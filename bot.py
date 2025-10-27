import asyncio
import json
import os
import re
import aiofiles
import numpy as np
from datetime import datetime, timedelta
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import TelegramError, NetworkError
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import logging
import hashlib
import signal
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Уменьшаем логи внешних библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.INFO)

# Настройка GPU для нейросетей
def setup_gpu():
    """Настройка использования GPU для нейросетей"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 Используется GPU: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("❌ GPU не доступен, используется CPU")
        return device
    except Exception as e:
        logger.error(f"Ошибка настройки GPU: {e}")
        return torch.device("cpu")

# Инициализация GPU
DEVICE = setup_gpu()

# Загрузка конфигурации
def load_config():
    try:
        with open('config/tg_config.json', 'r') as f:
            tg_config = json.load(f)
        with open('config/constants.json', 'r') as f:
            constants = json.load(f)
        
        # Загрузка конфигурации приватного канала-посредника
        with open('config/channel_config.json', 'r') as f:
            channel_config = json.load(f)
            
        return tg_config, constants, channel_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        default_constants = {
            "queue_ttl_seconds": 1800,
            "archive_ttl_days": 7,
            "queue_processing_interval": 300,
            "archive_cleanup_interval": 86400,
            "max_posts_per_batch": 5,
            "similarity_threshold": 0.85
        }
        default_channel_config = {
            "channel_id": -1001234567890
        }
        return {}, default_constants, default_channel_config

TG_CONFIG, CONSTANTS, CHANNEL_CONFIG = load_config()

class NeuralNewsProcessor:
    def __init__(self):
        logger.info("Загрузка нейросетевых моделей...")
        
        try:
            # Модель для эмбеддингов (русский язык) на GPU
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=str(DEVICE)
            )
            
            logger.info("✅ Нейросетевые модели загружены успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            raise

    def create_fingerprint(self, text):
        """Создание цифрового отпечатка текста"""
        if not text or len(text.strip()) < 10:
            return "0" * 64
            
        try:
            # Создаем эмбеддинг на GPU
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            
            # Конвертируем в хэш
            embedding_np = embedding.cpu().numpy()
            embedding_bytes = embedding_np.tobytes()
            return hashlib.sha256(embedding_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error creating fingerprint: {e}")
            return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def calculate_interest_score(self, text):
        """Оценка интересности текста"""
        if not text or len(text.strip()) < 20:
            return 0.0
        
        try:
            scores = []
            
            # 1. Оценка длины текста
            length_score = min(len(text) / 500, 1.0) * 0.3
            
            # 2. Оценка информативности через разнообразие слов
            words = text.split()
            if len(words) > 0:
                unique_words = set(words)
                diversity_score = len(unique_words) / len(words)
                scores.append(diversity_score * 0.3)
            
            # 3. Оценка структуры текста
            structure_score = self._calculate_structure_score(text)
            scores.append(structure_score * 0.4)
            
            total_score = length_score + sum(scores)
            return min(total_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating interest score: {e}")
            return 0.5

    def _calculate_structure_score(self, text):
        """Оценка структурного качества текста"""
        score = 0.0
        
        # Наличие чисел
        if any(char.isdigit() for char in text):
            score += 0.2
        
        # Наличие заглавных букв
        if any(char.isupper() for char in text):
            score += 0.2
        
        # Длина предложений
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 0:
            avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 20:
                score += 0.3
        
        # Наличие ключевых слов новостей
        news_keywords = ['новость', 'событие', 'сообщение', 'заявление', 'интервью', 
                        'анализ', 'данные', 'исследование', 'эксперт', 'официально']
        if any(keyword in text.lower() for keyword in news_keywords):
            score += 0.3
            
        return min(score, 1.0)

    def are_posts_similar(self, fingerprint1, fingerprint2):
        """Проверка схожести двух постов"""
        return fingerprint1 == fingerprint2

class ChannelMonitor:
    """Мониторинг каналов через пользовательский аккаунт"""
    
    def __init__(self, api_id, api_hash, neural_processor, bot_application):
        self.api_id = api_id
        self.api_hash = api_hash
        self.neural_processor = neural_processor
        self.bot_application = bot_application
        self.telethon_client = None
        self.is_running = False
        self.channel_handlers = {}
        self.monitored_channels = set()
        self.intermediate_channel_id = CHANNEL_CONFIG.get("channel_id")
        self.intermediate_channel_title = "Приватный канал-посредник"
        
    async def start(self):
        """Запуск мониторинга"""
        try:
            from telethon import TelegramClient, events
            
            logger.info("🔄 Запуск мониторинга каналов...")
            
            os.makedirs('session', exist_ok=True)

            self.telethon_client = TelegramClient(
                'session/user_monitor_session', 
                self.api_id, 
                self.api_hash
            )
            
            await self.telethon_client.start()
            
            me = await self.telethon_client.get_me()
            logger.info(f"✅ Мониторинг запущен от имени: {me.first_name} (@{me.username})")
            
            # Проверяем доступ к приватному каналу-посреднику
            await self._check_private_channel_access()
            
            self.is_running = True
            
            # Тестируем подключение к каналам
            await self._test_channel_connection()
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска мониторинга: {e}")
            raise

    async def _check_private_channel_access(self):
        """Проверка доступа к приватному каналу-посреднику"""
        try:
            if not self.intermediate_channel_id:
                logger.error("❌ Не указан ID приватного канала-посредника в config/channel_config.json")
                return False
                
            # Для приватных каналов используем только ID
            entity = await self.telethon_client.get_entity(self.intermediate_channel_id)
            channel_title = getattr(entity, 'title', 'Приватный канал')
            self.intermediate_channel_title = channel_title
            
            # Проверяем права бота в канале
            try:
                participant = await self.telethon_client.get_permissions(
                    entity, 
                    await self.telethon_client.get_me()
                )
                if participant.post_messages:
                    logger.info(f"✅ Доступ к приватному каналу подтвержден: {channel_title} (ID: {self.intermediate_channel_id})")
                    return True
                else:
                    logger.error("❌ Бот не имеет прав на отправку сообщений в приватный канал")
                    return False
            except:
                # Если не можем проверить права, но можем получить сущность - считаем что доступ есть
                logger.info(f"✅ Доступ к приватному каналу подтвержден: {channel_title} (ID: {self.intermediate_channel_id})")
                return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка доступа к приватному каналу: {e}")
            logger.error("Убедитесь, что:")
            logger.error("1. Приватный канал существует")
            logger.error("2. Бот добавлен в канал как администратор")
            logger.error("3. Указан правильный channel_id (отрицательное число)")
            logger.error("4. Бот имеет права на отправку сообщений")
            return False

    async def _test_channel_connection(self):
        """Тестирование подключения к каналам"""
        try:
            if not self.monitored_channels:
                logger.info("📭 Нет каналов для мониторинга")
                return
                
            logger.info(f"🔍 Тестируем подключение к {len(self.monitored_channels)} каналам...")
            
            for channel in list(self.monitored_channels)[:5]:
                try:
                    entity = await self.telethon_client.get_entity(channel)
                    logger.info(f"✅ Канал доступен: {channel} -> {getattr(entity, 'title', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось подключиться к каналу {channel}: {e}")
                    
        except Exception as e:
            logger.error(f"Ошибка тестирования каналов: {e}")
            
    async def stop(self):
        """Остановка мониторинга"""
        if self.telethon_client and self.telethon_client.is_connected():
            await self.telethon_client.disconnect()
        self.is_running = False
        logger.info("⏹️ Мониторинг каналов остановлен")
        
    async def add_channel_monitoring(self, chat_id, channels):
        """Добавление каналов для мониторинга чата"""
        try:
            from telethon import events
            
            if not self.telethon_client or not self.telethon_client.is_connected():
                await self.start()
            
            logger.info(f"📡 Добавляем каналы для чата {chat_id}: {channels}")
            
            # Добавляем каналы в общий список
            new_channels = []
            for channel in channels:
                if channel not in self.monitored_channels:
                    self.monitored_channels.add(channel)
                    new_channels.append(channel)
                    logger.info(f"   ➕ Новый канал: {channel}")
            
            if not new_channels:
                logger.info("   ℹ️ Все каналы уже отслеживаются")
                return
            
            # Удаляем старый обработчик если есть
            if chat_id in self.channel_handlers:
                self.telethon_client.remove_event_handler(self.channel_handlers[chat_id])
                logger.info(f"   🔄 Обновляем обработчик для чата {chat_id}")
            
            # Создаем новый обработчик для ВСЕХ отслеживаемых каналов
            @self.telethon_client.on(events.NewMessage(chats=list(self.monitored_channels)))
            async def message_handler(event):
                await self._process_new_post(chat_id, event.message)
            
            self.channel_handlers[chat_id] = message_handler
            
            logger.info(f"✅ Добавлены каналы для чата {chat_id}: {len(new_channels)} новых каналов")
            logger.info(f"📊 Всего отслеживаемых каналов: {len(self.monitored_channels)}")
            
            # Тестируем подключение к новым каналам
            for channel in new_channels:
                try:
                    entity = await self.telethon_client.get_entity(channel)
                    logger.info(f"🔗 Канал подключен: {channel} -> {getattr(entity, 'title', 'Unknown')}")
                except Exception as e:
                    logger.error(f"❌ Ошибка подключения к каналу {channel}: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка добавления каналов для чата {chat_id}: {e}")
            
    async def remove_channel_monitoring(self, chat_id, channels_to_remove):
        """Удаление каналов из мониторинга чата"""
        try:
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            if not os.path.exists(chat_data_path):
                return False
                
            chat_data = await self._safe_json_load(chat_data_path)
            if chat_data is None:
                return False
                
            current_channels = chat_data.get('channels', [])
            updated_channels = [ch for ch in current_channels if ch not in channels_to_remove]
            
            if len(updated_channels) == len(current_channels):
                return False  # Ничего не изменилось
            
            # Удаляем каналы из общего списка если они больше никем не используются
            for channel in channels_to_remove:
                if self._is_channel_used_by_others(chat_id, channel):
                    continue
                if channel in self.monitored_channels:
                    self.monitored_channels.remove(channel)
                    logger.info(f"   ➖ Удален канал: {channel}")
            
            chat_data['channels'] = updated_channels
            chat_data['updated_at'] = datetime.now().isoformat()
            
            await self._safe_json_save(chat_data_path, chat_data)
            
            # Перезагружаем обработчики с обновленным списком каналов
            if updated_channels:
                await self.add_channel_monitoring(chat_id, updated_channels)
            else:
                # Если каналов не осталось, удаляем обработчик
                if chat_id in self.channel_handlers:
                    self.telethon_client.remove_event_handler(self.channel_handlers[chat_id])
                    del self.channel_handlers[chat_id]
                    logger.info(f"   🗑️ Удален обработчик для чата {chat_id}")
            
            logger.info(f"✅ Удалены каналы для чата {chat_id}: {len(channels_to_remove)} каналов")
            logger.info(f"📊 Всего отслеживаемых каналов: {len(self.monitored_channels)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления каналов для чата {chat_id}: {e}")
            return False

    def _is_channel_used_by_others(self, current_chat_id, channel):
        """Проверяет, используется ли канал другими чатами"""
        try:
            if not os.path.exists('data/chats'):
                return False
                
            for chat_folder in os.listdir('data/chats'):
                if chat_folder == str(current_chat_id):
                    continue
                    
                chat_data_path = f"data/chats/{chat_folder}/chat_data.json"
                chat_data = self._safe_json_load_sync(chat_data_path)
                if chat_data and channel in chat_data.get('channels', []):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking channel usage: {e}")
            return False
            
    async def _process_new_post(self, chat_id, message):
        """Обработка нового поста"""
        try:
            logger.info(f"🎯 ПОЛУЧЕНО СООБЩЕНИЕ ИЗ КАНАЛА ДЛЯ ЧАТА {chat_id}")
            
            # Проверяем, активен ли бот для этого чата
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            chat_data = await self._safe_json_load(chat_data_path)
            if chat_data and not chat_data.get('is_active', True):
                logger.info(f"   ⏸️ Бот неактивен для чата {chat_id}, пропускаем сообщение")
                return
            
            # Пропускаем сообщения без текста (только медиа)
            if not message.text and not message.message:
                logger.info("   📭 Сообщение без текста (только медиа) - пропускаем")
                return
            
            # Получаем текст сообщения
            message_text = message.text or message.message or ""
            logger.info(f"   📝 Текст сообщения: {message_text[:100]}...")
            
            if len(message_text.strip()) < 10:
                logger.info(f"   📏 Слишком короткое сообщение ({len(message_text.strip())} chars) - пропускаем")
                return
            
            chat = await message.get_chat()
            channel_username = getattr(chat, 'username', None)
            channel_title = getattr(chat, 'title', 'Unknown Channel')
            
            logger.info(f"   📢 Канал: {channel_title} (@{channel_username})")
            logger.info(f"   🆔 ID сообщения: {message.id}")
            logger.info(f"   📏 Длина текста: {len(message_text)} символов")
            
            # Создание отпечатка и оценка интересности
            logger.info("   🧠 Анализируем сообщение нейросетью...")
            fingerprint = self.neural_processor.create_fingerprint(message_text)
            interest_score = self.neural_processor.calculate_interest_score(message_text)
            
            # Сохраняем метаданные для пересылки через приватный канал-посредник
            post_data = {
                'id': message.id,
                'channel': channel_username if channel_username else channel_title,
                'channel_id': chat.id,
                'message_id': message.id,
                'timestamp': datetime.now().isoformat(),
                'url': f"https://t.me/{channel_username}/{message.id}" if channel_username else f"https://t.me/c/{str(chat.id).replace('-100', '')}/{message.id}",
                'has_media': bool(message.media),
                'is_forward': bool(message.forward),
                'chat_id': chat_id,
                'fingerprint': fingerprint,
                'interest_score': interest_score,
                'original_message_id': message.id,
                'original_channel_id': chat.id,
                'message_object': None  # Не сохраняем объект сообщения для безопасности
            }
            
            logger.info(f"   🔑 Отпечаток: {fingerprint[:16]}...")
            logger.info(f"   ⭐ Оценка интересности: {interest_score:.2f}/1.0")
            
            await self._add_to_chat_queue(chat_id, post_data)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки поста для чата {chat_id}: {e}")
            
    async def _add_to_chat_queue(self, chat_id, post_data):
        """Добавление поста в очередь чата"""
        try:
            logger.info(f"   📥 Добавляем пост в очередь чата {chat_id}...")
            
            queue_path = f"data/chats/{chat_id}/queue.json"
            archive_path = f"data/chats/{chat_id}/archive.json"
            
            os.makedirs(os.path.dirname(queue_path), exist_ok=True)
            
            # Безопасная загрузка JSON
            queue = await self._safe_json_load(queue_path) or []
            archive = await self._safe_json_load(archive_path) or []
            
            logger.info(f"   📊 Текущий размер очереди: {len(queue)} постов")
            
            archive_fingerprints = {item.get('fingerprint') for item in archive if item.get('fingerprint')}
            if post_data['fingerprint'] in archive_fingerprints:
                logger.info(f"   📭 Пост уже в архиве, пропускаем: {post_data['fingerprint'][:16]}...")
                return
            
            duplicate_index = None
            for i, queued_post in enumerate(queue):
                if self.neural_processor.are_posts_similar(queued_post.get('fingerprint'), post_data['fingerprint']):
                    duplicate_index = i
                    logger.info(f"   🔄 Найден дубликат поста в позиции {i}")
                    break
            
            if duplicate_index is not None:
                if post_data['interest_score'] > queue[duplicate_index]['interest_score']:
                    queue[duplicate_index] = post_data
                    logger.info(f"   ✅ Заменен дубликат поста для чата {chat_id}")
                else:
                    logger.info(f"   📭 Дубликат имеет лучшую оценку, пропускаем")
            else:
                queue.append(post_data)
                logger.info(f"   ✅ Добавлен новый пост в очередь для чата {chat_id}")
            
            # Безопасное сохранение JSON
            await self._safe_json_save(queue_path, queue)
            logger.info(f"   💾 Очередь сохранена, новый размер: {len(queue)} постов")
                
            await self._update_chat_stats(chat_id, 'processed')
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в очередь для чата {chat_id}: {e}")
    
    async def _safe_json_load(self, filepath):
        """Безопасная загрузка JSON файла (асинхронная версия)"""
        try:
            if os.path.exists(filepath):
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        return json.loads(content)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка чтения JSON файла {filepath}: {e}")
            backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.exists(filepath):
                os.rename(filepath, backup_path)
                logger.info(f"📦 Создана резервная копия поврежденного файла: {backup_path}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки файла {filepath}: {e}")
            return None

    def _safe_json_load_sync(self, filepath):
        """Безопасная загрузка JSON файла (синхронная версия)"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        return json.loads(content)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка чтения JSON файла {filepath}: {e}")
            backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.exists(filepath):
                os.rename(filepath, backup_path)
                logger.info(f"📦 Создана резервная копия поврежденного файла: {backup_path}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки файла {filepath}: {e}")
            return None
    
    async def _safe_json_save(self, filepath, data):
        """Безопасное сохранение JSON файла"""
        try:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения файла {filepath}: {e}")
            
    async def _update_chat_stats(self, chat_id, stat_type):
        """Обновление статистики чата"""
        try:
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            
            chat_data = await self._safe_json_load(chat_data_path)
            if chat_data is None:
                chat_data = {
                    'channels': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_processed': 0,
                    'total_sent': 0,
                    'chat_type': 'unknown',
                    'is_active': True
                }
            
            if stat_type == 'processed':
                chat_data['total_processed'] = chat_data.get('total_processed', 0) + 1
            
            chat_data['updated_at'] = datetime.now().isoformat()
            
            await self._safe_json_save(chat_data_path, chat_data)
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статистики для чата {chat_id}: {e}")

class NewsBot:
    def __init__(self):
        self.bot_token = TG_CONFIG.get('bot_token', '')
        self.api_id = TG_CONFIG.get('api_id', 0)
        self.api_hash = TG_CONFIG.get('api_hash', '')
        
        if not all([self.bot_token, self.api_id, self.api_hash]):
            logger.error("❌ Не заполнены конфигурационные данные в config/tg_config.json")
            sys.exit(1)
            
        # Создаем Application с настройками для обработки сетевых ошибок
        self.application = (
            Application.builder()
            .token(self.bot_token)
            .pool_timeout(30)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .build()
        )
        
        self.neural_processor = NeuralNewsProcessor()
        self.channel_monitor = ChannelMonitor(self.api_id, self.api_hash, self.neural_processor, self.application)
        
        self.setup_handlers()
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Обработка сигналов для graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Получен сигнал завершения...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def setup_handlers(self):
        """Настройка обработчиков команд с кнопками"""
        # Основные команды
        self.application.add_handler(CommandHandler("start", self.start_handler))
        self.application.add_handler(CommandHandler("add_channels", self.add_channels_handler))
        self.application.add_handler(CommandHandler("my_channels", self.my_channels_handler))
        self.application.add_handler(CommandHandler("remove_channels", self.remove_channels_handler))
        self.application.add_handler(CommandHandler("stop", self.stop_handler))
        self.application.add_handler(CommandHandler("help", self.help_handler))
        
        # Обработчик callback query ДОЛЖЕН БЫТЬ ПЕРВЫМ среди callback handlers
        self.application.add_handler(CallbackQueryHandler(self.callback_handler, pattern="^rm_"))
        
        # Обработчик текстовых сообщений
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
        
        # Обработчик ошибок
        self.application.add_error_handler(self.error_handler)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок с фильтрацией сообщений от бота"""
        try:
            # Пропускаем ошибки от пересланных сообщений
            if update and update.effective_message:
                # Безопасная проверка forward_from_chat
                if (hasattr(update.effective_message, 'forward_from_chat') and 
                    update.effective_message.forward_from_chat and 
                    update.effective_message.forward_from_chat.id == self.channel_monitor.intermediate_channel_id):
                    logger.debug("🔇 Игнорируем ошибку от пересланного сообщения")
                    return
                    
                # Если сообщение от самого бота - игнорируем
                if (update.effective_message.from_user and 
                    update.effective_message.from_user.id == self.application.bot.id):
                    logger.debug("🔇 Игнорируем ошибку от сообщения бота")
                    return

            logger.error(f"Exception while handling an update: {context.error}")
            
            if isinstance(context.error, NetworkError):
                logger.warning(f"Network error occurred: {context.error}")
                return
            
            logger.error(f"Traceback: {context.error.__traceback__}")
            
            # Отправляем сообщение об ошибке только если это не пересланное сообщение
            if update and update.effective_chat:
                try:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ Произошла ошибка при обработке запроса. Попробуйте еще раз."
                    )
                except Exception as e:
                    logger.error(f"Error sending error message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
    
    def _create_channels_keyboard(self, channels, selected_indices):
        """Создает клавиатуру с каналами в 2 столбца"""
        keyboard = []
        
        # Создаем кнопки каналов в 2 столбца
        for i in range(0, len(channels), 2):
            row = []
            for j in range(2):
                if i + j < len(channels):
                    channel_index = i + j
                    channel = channels[channel_index]
                    
                    # Обрезаем длинные названия
                    display_name = channel
                    if len(channel) > 15:
                        display_name = channel[:12] + "..."
                    
                    # Определяем эмодзи
                    emoji = "❌" if channel_index in selected_indices else "✅"
                    
                    button = InlineKeyboardButton(
                        f"{emoji} {display_name}",
                        callback_data=f"rm_{channel_index}"  # Короткий префикс
                    )
                    row.append(button)
            keyboard.append(row)
        
        # Добавляем кнопки действий
        action_buttons = []
        if selected_indices:
            action_buttons.append(InlineKeyboardButton("🚀 Подтвердить", callback_data="rm_confirm"))
        action_buttons.append(InlineKeyboardButton("❌ Отмена", callback_data="rm_cancel"))
        keyboard.append(action_buttons)
        
        return InlineKeyboardMarkup(keyboard)

    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback query для удаления каналов"""
        query = update.callback_query
        await query.answer()
        
        chat_id = update.effective_chat.id
        data = query.data
        
        logger.info(f"📨 Callback получен: {data} для чата {chat_id}")
        
        # Обрабатываем только callback'и связанные с удалением каналов
        if not data.startswith("rm_"):
            return
        
        try:
            if data == "rm_confirm":
                await self._handle_confirm_remove(query, context, chat_id)
                
            elif data == "rm_cancel":
                await self._handle_cancel_remove(query, context)
                
            elif data.startswith("rm_"):
                # Извлекаем индекс из callback data
                index_str = data[3:]  # Убираем "rm_"
                if index_str.isdigit():
                    await self._handle_toggle_channel(query, context, chat_id, int(index_str))
                else:
                    await query.answer("❌ Ошибка: неверный формат данных", show_alert=True)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка в callback_handler: {e}")
            await self._handle_callback_error(query, context, chat_id)

    async def _handle_toggle_channel(self, query, context, chat_id, channel_index):
        """Обработка переключения выбора канала"""
        remove_data = context.chat_data.get('remove_channels')
        if not remove_data:
            await query.answer("❌ Сессия устарела. Начните заново.", show_alert=True)
            return
        
        channels = remove_data['available_channels']
        selected_indices = remove_data['selected_indices']
        
        # Проверяем валидность индекса
        if channel_index < 0 or channel_index >= len(channels):
            await query.answer("❌ Ошибка: неверный индекс канала", show_alert=True)
            return
        
        # Переключаем выбор
        if channel_index in selected_indices:
            selected_indices.remove(channel_index)
        else:
            selected_indices.append(channel_index)
        
        # Обновляем клавиатуру
        keyboard = self._create_channels_keyboard(channels, selected_indices)
        
        try:
            await query.edit_message_text(
                f"🗑️ **Удаление каналов**\n\n"
                f"✅ - канал отслеживается\n"
                f"❌ - канал выбран для удаления\n\n"
                f"Выберите каналы для удаления:\n\n"
                f"📊 Выбрано: {len(selected_indices)}/{len(channels)}",
                reply_markup=keyboard
            )
        except Exception as e:
            logger.error(f"❌ Ошибка обновления сообщения: {e}")
            await query.answer("❌ Ошибка обновления", show_alert=True)

    async def _handle_confirm_remove(self, query, context, chat_id):
        """Обработка подтверждения удаления"""
        try:
            remove_data = context.chat_data.get('remove_channels')
            if not remove_data:
                await query.answer("❌ Сессия устарела. Начните заново.", show_alert=True)
                return
            
            selected_indices = remove_data['selected_indices']
            channels = remove_data['available_channels']
            
            if not selected_indices:
                await query.answer("❌ Не выбрано ни одного канала", show_alert=True)
                return
            
            # Получаем имена каналов для удаления
            channels_to_remove = [channels[i] for i in selected_indices if i < len(channels)]
            
            if not channels_to_remove:
                await query.answer("❌ Ошибка: каналы не найдены", show_alert=True)
                return
            
            # Убираем клавиатуру сразу, показываем процесс
            empty_keyboard = InlineKeyboardMarkup([])
            await query.edit_message_text(
                "🔄 Удаляем выбранные каналы...",
                reply_markup=empty_keyboard
            )
            
            # Удаляем каналы
            success = await self.channel_monitor.remove_channel_monitoring(chat_id, channels_to_remove)
            
            if success:
                channels_list = "\n".join([f"• {channel}" for channel in channels_to_remove])
                await query.edit_message_text(
                    f"✅ **Каналы успешно удалены:**\n\n{channels_list}\n\n"
                    f"Теперь эти каналы больше не отслеживаются."
                )
            else:
                await query.edit_message_text(
                    "❌ **Ошибка при удалении каналов**\n\n"
                    "Попробуйте еще раз или обратитесь к администратору."
                )
            
            # Очищаем временные данные
            if 'remove_channels' in context.chat_data:
                del context.chat_data['remove_channels']
                
        except Exception as e:
            logger.error(f"❌ Ошибка подтверждения удаления: {e}")
            await query.edit_message_text(
                "❌ **Произошла ошибка при удалении каналов**\n\n"
                "Попробуйте еще раз или обратитесь к администратору."
            )

    async def _handle_cancel_remove(self, query, context):
        """Обработка отмены удаления"""
        try:
            # Очищаем временные данные
            if 'remove_channels' in context.chat_data:
                del context.chat_data['remove_channels']
            
            await query.edit_message_text("❌ Удаление каналов отменено.")
            
        except Exception as e:
            logger.error(f"❌ Ошибка отмены удаления: {e}")
            await query.answer("❌ Ошибка отмены", show_alert=True)

    async def _handle_callback_error(self, query, context, chat_id):
        """Обработка ошибок callback"""
        try:
            await query.edit_message_text(
                "❌ **Произошла ошибка**\n\n"
                "Попробуйте выполнить команду заново."
            )
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения об ошибке: {e}")
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Произошла ошибка. Попробуйте выполнить команду заново."
                )
            except Exception as e2:
                logger.error(f"❌ Критическая ошибка отправки сообщения: {e2}")

    def get_main_keyboard(self):
        """Клавиатура с основными командами"""
        keyboard = [
            [KeyboardButton("/add_channels"), KeyboardButton("/my_channels")],
            [KeyboardButton("/remove_channels"), KeyboardButton("/stop")],
            [KeyboardButton("/help")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, input_field_placeholder="Выберите команду...")
    
    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type
        
        chat_type_str = "личные сообщения" if chat_type == "private" else f"группа '{update.effective_chat.title}'"
        
        chat_folder = f"data/chats/{chat_id}"
        os.makedirs(chat_folder, exist_ok=True)
        
        await self.create_chat_files(chat_id, chat_type)
        
        welcome_text = f"""
🤖 Добро пожаловать в All News Bot!

💬 **Режим работы:** {chat_type_str}

🎯 **Новые возможности:**
• 📨 Пересылка оригинальных сообщений через ПРИВАТНЫЙ канал-посредник
• 👥 Работа в группах и личных сообщениях
• 🔗 Сохранение форматирования и медиа
• 🧠 Умная фильтрация дубликатов
• 📢 Сообщения пересылаются от имени приватного канала
• 🔒 Максимальная конфиденциальность

📋 **Основные команды:**
• /add_channels - добавить каналы
• /my_channels - мои каналы  
• /remove_channels - удалить каналы
• /stop - остановить отправку новостей
• /help - помощь

💡 **Бот теперь пересылает оригинальные сообщения через ПРИВАТНЫЙ канал-посредник!**
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=self.get_main_keyboard()
        )
    
    async def stop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Остановка отправки новостей"""
        chat_id = update.effective_chat.id
        
        try:
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            chat_data = await self.channel_monitor._safe_json_load(chat_data_path)
            
            if chat_data:
                chat_data['is_active'] = False
                chat_data['updated_at'] = datetime.now().isoformat()
                await self.channel_monitor._safe_json_save(chat_data_path, chat_data)
                
                await update.message.reply_text(
                    "⏸️ **Бот остановлен**\n\n"
                    "Я больше не буду присылать новости в этот чат.\n"
                    "Чтобы возобновить получение новостей, используйте команду /start\n\n"
                    "ℹ️ Все ваши настройки и каналы сохранены.",
                    reply_markup=self.get_main_keyboard()
                )
                logger.info(f"🛑 Бот остановлен для чата {chat_id}")
            else:
                await update.message.reply_text(
                    "❌ **Чат не найден**\n\n"
                    "Сначала запустите бота с помощью /start",
                    reply_markup=self.get_main_keyboard()
                )
                
        except Exception as e:
            logger.error(f"❌ Ошибка остановки бота для чата {chat_id}: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при остановке бота. Попробуйте еще раз.",
                reply_markup=self.get_main_keyboard()
            )
    
    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка по командам"""
        help_text = """
📋 **Доступные команды:**

🔍 **Управление каналами:**
`/add_channels` - добавить каналы для отслеживания
`/my_channels` - показать мои каналы
`/remove_channels` - удалить каналы из отслеживания
`/stop` - остановить отправку новостей

❓ **Помощь:**
`/help` - показать это сообщение
`/start` - перезапустить бота

🔄 **Новый функционал:**
• 📨 Пересылка оригинальных сообщений через ПРИВАТНЫЙ канал
• 👥 Работа в группах
• 🔗 Сохранение медиа и форматирования
• 🧠 Умная фильтрация дубликатов
• 📢 Сообщения от имени приватного канала-посредника
• 🔒 Максимальная конфиденциальность

💡 **Формат каналов:**
t.me/*channel_name*
@*username*
https://t.me/*channel*
        """
        
        await update.message.reply_text(
            help_text,
            reply_markup=self.get_main_keyboard()
        )
    
    async def add_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды добавления каналов"""
        chat_id = update.effective_chat.id
        
        if 'chat_data' not in context.chat_data:
            context.chat_data['chat_data'] = {}
        context.chat_data['chat_data']['awaiting_channels'] = True
        
        logger.info(f"🟢 Установлен флаг awaiting_channels для чата {chat_id}")
        
        await update.message.reply_text(
            "📥 **Добавление каналов**\n\n"
            "Пришлите ссылки на Telegram каналы (каждую с новой строки):\n\n"
            "**Примеры:**\n"
            "t.me/rbc_news\n"
            "@meduzaproject\n"
            "https://t.me/rian_ru\n\n"
            "🎯 Бот будет пересылать оригинальные сообщения из этих каналов через ПРИВАТНЫЙ канал-посредник!",
            reply_markup=self.get_main_keyboard()
        )

    async def my_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать отслеживаемые каналы"""
        chat_id = update.effective_chat.id
        
        chat_data = await self.channel_monitor._safe_json_load(f"data/chats/{chat_id}/chat_data.json")
        
        if chat_data and chat_data.get('channels'):
            channels = chat_data.get('channels', [])
            channels_text = "\n".join([f"• {channel}" for channel in channels])
            
            status = "🟢 Активен" if chat_data.get('is_active', True) else "⏸️ Остановлен"
            
            message = f"📋 **Ваши отслеживаемые каналы** ({len(channels)}):\n\n{channels_text}\n\n**Статус бота:** {status}\n\n💡 Используйте /remove_channels чтобы удалить каналы"
        else:
            message = "❌ У вас нет отслеживаемых каналов.\n\n💡 Добавьте каналы командой /add_channels"
        
        await update.message.reply_text(
            message,
            reply_markup=self.get_main_keyboard()
        )

    async def remove_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Удаление каналов из отслеживания с выбором и подтверждением"""
        chat_id = update.effective_chat.id
        
        try:
            chat_data = await self.channel_monitor._safe_json_load(f"data/chats/{chat_id}/chat_data.json")
            if not chat_data or not chat_data.get('channels'):
                await update.message.reply_text(
                    "❌ У вас нет отслеживаемых каналов.",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            channels = chat_data.get('channels', [])
            
            # Сохраняем каналы в context с индексами
            context.chat_data['remove_channels'] = {
                'available_channels': channels,
                'selected_indices': []
            }
            
            # Создаем начальную клавиатуру
            keyboard = self._create_channels_keyboard(channels, [])
            
            message = await update.message.reply_text(
                "🗑️ **Удаление каналов**\n\n"
                "✅ - канал отслеживается\n"
                "❌ - канал выбран для удаления\n\n"
                "Выберите каналы для удаления (нажмите на кнопку чтобы выбрать/отменить выбор):"
                f"\n\n📊 Выбрано: 0/{len(channels)}",
                reply_markup=keyboard
            )
            
            # Сохраняем ID сообщения для последующих обновлений
            context.chat_data['remove_message_id'] = message.message_id
            
        except Exception as e:
            logger.error(f"❌ Ошибка в remove_channels_handler: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при загрузке каналов",
                reply_markup=self.get_main_keyboard()
            )

    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        chat_id = update.effective_chat.id
        message_text = update.message.text
        
        logger.info(f"📨 Получено сообщение в чате {chat_id}: {message_text[:100]}...")
        
        chat_data = context.chat_data.get('chat_data', {})
        if chat_data.get('awaiting_channels'):
            logger.info(f"🟢 Обрабатываем ввод каналов для чата {chat_id}")
            
            context.chat_data['chat_data']['awaiting_channels'] = False
            
            await self.process_channels_input(update, message_text)
            return
        
        await update.message.reply_text(
            "🤖 Используйте команды из меню или /help для справки.",
            reply_markup=self.get_main_keyboard()
        )

    async def process_channels_input(self, update: Update, message_text: str):
        """Обработка введенных каналов"""
        chat_id = update.effective_chat.id
        
        try:
            raw_channels = [line.strip() for line in message_text.split('\n') if line.strip()]
            
            if not raw_channels:
                await update.message.reply_text(
                    "❌ Не найдено валидных каналов. Попробуйте еще раз.",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            processed_channels = []
            invalid_channels = []
            
            for channel in raw_channels:
                processed_channel = self.process_channel_input(channel)
                if processed_channel:
                    processed_channels.append(processed_channel)
                else:
                    invalid_channels.append(channel)
            
            if not processed_channels:
                await update.message.reply_text(
                    "❌ Не найдено валидных каналов. Проверьте формат ввода.",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            await self.save_channels_for_chat(chat_id, processed_channels)
            await self.channel_monitor.add_channel_monitoring(chat_id, processed_channels)
            
            success_text = f"✅ **Добавлено каналов:** {len(processed_channels)}\n\n"
            success_text += "\n".join([f"• {channel}" for channel in processed_channels])
            
            if invalid_channels:
                success_text += f"\n\n❌ **Невалидные каналы:** {len(invalid_channels)}\n"
                success_text += "\n".join([f"• {channel}" for channel in invalid_channels])
            
            success_text += f"\n\n🎯 Теперь бот будет пересылать сообщения из этих каналов через ПРИВАТНЫЙ канал-посредник!"
            
            await update.message.reply_text(
                success_text,
                reply_markup=self.get_main_keyboard()
            )
            
        except Exception as e:
            logger.error(f"Ошибка обработки каналов для чата {chat_id}: {e}")
            await update.message.reply_text(
                f"❌ Ошибка обработки каналов: {e}",
                reply_markup=self.get_main_keyboard()
            )

    def process_channel_input(self, channel_input: str) -> str:
        """Обработка одного канала"""
        channel_input = channel_input.strip()
        
        if channel_input.startswith('https://t.me/'):
            channel_input = channel_input.replace('https://t.me/', '')
        elif channel_input.startswith('t.me/'):
            channel_input = channel_input.replace('t.me/', '')
        
        if channel_input.startswith('/'):
            channel_input = channel_input[1:]
        
        if channel_input.startswith('@'):
            channel_input = channel_input[1:]
        
        if not channel_input or len(channel_input) < 3:
            return None
        
        if any(char in channel_input for char in [' ', '/', '\\', '?', '#']):
            return None
        
        return f"@{channel_input}" if not channel_input.startswith('@') else channel_input

    async def save_channels_for_chat(self, chat_id, channels):
        """Сохранение каналов для чата"""
        try:
            chat_folder = f"data/chats/{chat_id}"
            os.makedirs(chat_folder, exist_ok=True)
            
            chat_data_path = f"{chat_folder}/chat_data.json"
            
            chat_data = await self.channel_monitor._safe_json_load(chat_data_path)
            if chat_data is None:
                chat_data = {
                    'channels': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_processed': 0,
                    'total_sent': 0,
                    'chat_type': 'private',
                    'is_active': True
                }
            
            existing_channels = set(chat_data['channels'])
            new_channels = [ch for ch in channels if ch not in existing_channels]
            
            chat_data['channels'].extend(new_channels)
            chat_data['updated_at'] = datetime.now().isoformat()
            
            await self.channel_monitor._safe_json_save(chat_data_path, chat_data)
            
            logger.info(f"✅ Сохранено {len(new_channels)} новых каналов для чата {chat_id}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения каналов для чата {chat_id}: {e}")
            raise

    async def create_chat_files(self, chat_id, chat_type):
        """Создание файлов для нового чата"""
        try:
            chat_folder = f"data/chats/{chat_id}"
            os.makedirs(chat_folder, exist_ok=True)
            
            chat_data_path = f"{chat_folder}/chat_data.json"
            queue_path = f"{chat_folder}/queue.json"
            archive_path = f"{chat_folder}/archive.json"
            
            if not os.path.exists(chat_data_path):
                chat_data = {
                    'channels': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_processed': 0,
                    'total_sent': 0,
                    'chat_type': chat_type,
                    'is_active': True
                }
                await self.channel_monitor._safe_json_save(chat_data_path, chat_data)
            
            if not os.path.exists(queue_path):
                await self.channel_monitor._safe_json_save(queue_path, [])
            
            if not os.path.exists(archive_path):
                await self.channel_monitor._safe_json_save(archive_path, [])
                
            logger.info(f"✅ Созданы файлы для чата {chat_id}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания файлов для чата {chat_id}: {e}")

    async def restore_channel_monitoring(self):
        """Восстановление отслеживания каналов после перезагрузки"""
        try:
            logger.info("🔄 Восстанавливаю отслеживание каналов для всех чатов...")
            
            if not os.path.exists('data/chats'):
                logger.info("📁 Нет данных чатов для восстановления")
                return
                
            chat_folders = os.listdir('data/chats')
            total_channels = 0
            
            for chat_folder in chat_folders:
                try:
                    chat_id = int(chat_folder)
                    chat_data_path = f"data/chats/{chat_folder}/chat_data.json"
                    
                    chat_data = await self.channel_monitor._safe_json_load(chat_data_path)
                    if chat_data and chat_data.get('channels'):
                        channels = chat_data['channels']
                        if channels:
                            await self.channel_monitor.add_channel_monitoring(chat_id, channels)
                            total_channels += len(channels)
                            logger.info(f"   ✅ Восстановлено {len(channels)} каналов для чата {chat_id}")
                            
                except Exception as e:
                    logger.error(f"❌ Ошибка восстановления каналов для чата {chat_folder}: {e}")
                    continue
                    
            logger.info(f"✅ Восстановление завершено: {total_channels} каналов для {len(chat_folders)} чатов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при восстановлении отслеживания каналов: {e}")

    async def start_monitoring(self):
        """Запуск мониторинга каналов"""
        try:
            await self.channel_monitor.start()
            logger.info("✅ Мониторинг каналов запущен")
            
            await self.restore_channel_monitoring()
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска мониторинга: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Завершение работы бота...")
        
        await self.channel_monitor.stop()
        
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
        
        logger.info("👋 Бот завершил работу")

    async def process_queue_loop(self):
        """Цикл обработки очереди"""
        while True:
            try:
                await self.process_all_queues()
                await asyncio.sleep(CONSTANTS['queue_processing_interval'])
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле обработки очереди: {e}")
                await asyncio.sleep(60)

    async def process_all_queues(self):
        """Обработка всех очередей чатов"""
        try:
            if not os.path.exists('data/chats'):
                return
            
            for chat_folder in os.listdir('data/chats'):
                chat_id = chat_folder
                queue_path = f"data/chats/{chat_folder}/queue.json"
                
                if os.path.exists(queue_path):
                    await self.process_chat_queue(chat_id)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка обработки всех очередей: {e}")

    async def process_chat_queue(self, chat_id: str):
        """Обработка очереди конкретного чата"""
        try:
            queue_path = f"data/chats/{chat_id}/queue.json"
            archive_path = f"data/chats/{chat_id}/archive.json"
            
            queue = await self.channel_monitor._safe_json_load(queue_path) or []
            archive = await self.channel_monitor._safe_json_load(archive_path) or []
            
            if not queue:
                return
            
            # Проверяем активность бота для этого чата
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            chat_data = await self.channel_monitor._safe_json_load(chat_data_path)
            if chat_data and not chat_data.get('is_active', True):
                logger.info(f"⏸️ Бот неактивен для чата {chat_id}, пропускаем обработку очереди")
                return
            
            queue.sort(key=lambda x: x.get('interest_score', 0), reverse=True)
            
            top_posts = queue[:CONSTANTS['max_posts_per_batch']]
            
            sent_count = 0
            for post in top_posts:
                try:
                    success = await self.forward_via_private_channel(int(chat_id), post)
                    
                    if success:
                        archive.append(post)
                        sent_count += 1
                        logger.info(f"✅ Переслан пост в чат {chat_id} через приватный канал")
                        
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"⚠️ Не удалось переслать пост в чат {chat_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка пересылки поста в чат {chat_id}: {e}")
            
            remaining_queue = queue[CONSTANTS['max_posts_per_batch']:]
            
            await self.channel_monitor._safe_json_save(queue_path, remaining_queue)
            await self.channel_monitor._safe_json_save(archive_path, archive)
            
            if sent_count > 0:
                await self.update_chat_sent_stats(int(chat_id), sent_count)
            
            logger.info(f"📤 Обработана очередь чата {chat_id}: отправлено {sent_count} постов, осталось {len(remaining_queue)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки очереди чата {chat_id}: {e}")

    async def forward_via_private_channel(self, chat_id: int, post_data: dict) -> bool:
        """Пересылка сообщения через приватный канал-посредник с улучшенной обработкой ошибок"""
        try:
            logger.info(f"🔄 Начинаем пересылку для чата {chat_id} через приватный канал...")
            
            if not self.channel_monitor.telethon_client or not self.channel_monitor.telethon_client.is_connected():
                logger.error("❌ Telethon клиент не подключен")
                return False

            # Проверяем доступ к приватному каналу
            try:
                logger.info(f"🔍 Проверяем доступ к приватному каналу {self.channel_monitor.intermediate_channel_id}...")
                channel_entity = await self.channel_monitor.telethon_client.get_entity(
                    self.channel_monitor.intermediate_channel_id
                )
                logger.info(f"✅ Доступ к каналу подтвержден: {getattr(channel_entity, 'title', 'Unknown')}")
            except Exception as e:
                logger.error(f"❌ Ошибка доступа к приватному каналу: {e}")
                logger.error("Убедитесь, что:")
                logger.error("1. Канал существует и является приватным")
                logger.error("2. Бот добавлен в канал как администратор")
                logger.error("3. Указан правильный channel_id")
                return False

            # Получаем оригинальное сообщение
            logger.info(f"📨 Получаем оригинальное сообщение {post_data['original_message_id']}...")
            original_message = await self.channel_monitor.telethon_client.get_messages(
                post_data['original_channel_id'],
                ids=post_data['original_message_id']
            )
            
            if not original_message:
                logger.error(f"❌ Не удалось получить сообщение {post_data['original_message_id']} из канала {post_data['original_channel_id']}")
                return False

            logger.info("✅ Оригинальное сообщение получено")

            # Пересылаем сообщение в приватный канал-посредник
            logger.info("🔄 Пересылаем сообщение в приватный канал...")
            try:
                forwarded_message = await self.channel_monitor.telethon_client.forward_messages(
                    entity=self.channel_monitor.intermediate_channel_id,
                    messages=original_message,
                    from_peer=post_data['original_channel_id']
                )
                
                if not forwarded_message:
                    logger.error("❌ Не удалось переслать сообщение в приватный канал")
                    return False
                    
                logger.info("✅ Сообщение переслано в приватный канал")

            except Exception as e:
                logger.error(f"❌ Ошибка пересылки в приватный канал: {e}")
                logger.error("Проверьте права бота в приватном канале")
                return False

            # Получаем ID пересланного сообщения в приватном канале
            if hasattr(forwarded_message, 'id'):
                intermediate_message_id = forwarded_message.id
            elif isinstance(forwarded_message, list) and len(forwarded_message) > 0:
                intermediate_message_id = forwarded_message[0].id
            else:
                logger.error("❌ Не удалось получить ID пересланного сообщения")
                return False

            logger.info(f"📝 ID сообщения в приватном канале: {intermediate_message_id}")

            # Пересылаем из приватного канала в целевой чат через Bot API
            logger.info(f"🔄 Пересылаем из приватного канала в чат {chat_id}...")
            try:
                await self.application.bot.forward_message(
                    chat_id=chat_id,
                    from_chat_id=self.channel_monitor.intermediate_channel_id,
                    message_id=intermediate_message_id
                )
                
                logger.info(f"✅ Сообщение успешно переслано через приватный канал в чат {chat_id}")
                return True

            except Exception as e:
                logger.error(f"❌ Ошибка пересылки в чат {chat_id}: {e}")
                logger.error("Проверьте:")
                logger.error(f"1. Бот добавлен в чат {chat_id}")
                logger.error("2. Бот имеет права на отправку сообщений в чат")
                return False

        except Exception as e:
            logger.error(f"❌ Критическая ошибка пересылки через приватный канал: {e}")
            return False

    async def update_chat_sent_stats(self, chat_id: int, sent_count: int):
        """Обновление статистики отправленных постов"""
        try:
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            
            chat_data = await self.channel_monitor._safe_json_load(chat_data_path)
            if chat_data is None:
                return
            
            chat_data['total_sent'] = chat_data.get('total_sent', 0) + sent_count
            chat_data['updated_at'] = datetime.now().isoformat()
            
            await self.channel_monitor._safe_json_save(chat_data_path, chat_data)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статистики отправки для чата {chat_id}: {e}")

    async def cleanup_archive_loop(self):
        """Цикл очистки архива"""
        while True:
            try:
                await self.cleanup_all_archives()
                await asyncio.sleep(CONSTANTS['archive_cleanup_interval'])
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле очистки архива: {e}")
                await asyncio.sleep(3600)

    async def cleanup_all_archives(self):
        """Очистка архивов всех чатов"""
        try:
            if not os.path.exists('data/chats'):
                return
            
            cutoff_time = datetime.now() - timedelta(days=CONSTANTS['archive_ttl_days'])
            
            for chat_folder in os.listdir('data/chats'):
                archive_path = f"data/chats/{chat_folder}/archive.json"
                
                if os.path.exists(archive_path):
                    await self.cleanup_chat_archive(chat_folder, archive_path, cutoff_time)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка очистки архивов: {e}")

    async def cleanup_chat_archive(self, chat_id: str, archive_path: str, cutoff_time: datetime):
        """Очистка архива конкретного чата"""
        try:
            archive = await self.channel_monitor._safe_json_load(archive_path) or []
            
            if not archive:
                return
            
            cleaned_archive = []
            removed_count = 0
            
            for post in archive:
                try:
                    post_time = datetime.fromisoformat(post.get('timestamp', '2000-01-01'))
                    if post_time > cutoff_time:
                        cleaned_archive.append(post)
                    else:
                        removed_count += 1
                except:
                    cleaned_archive.append(post)
            
            if removed_count > 0:
                await self.channel_monitor._safe_json_save(archive_path, cleaned_archive)
                logger.info(f"🧹 Очищен архив чата {chat_id}: удалено {removed_count} старых постов")
                
        except Exception as e:
            logger.error(f"❌ Ошибка очистки архива чата {chat_id}: {e}")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запуск All News Bot...")
            
            await self.start_monitoring()
            
            asyncio.create_task(self.process_queue_loop())
            asyncio.create_task(self.cleanup_archive_loop())
            
            logger.info("🤖 Бот запускается...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
            logger.info("✅ Бот успешно запущен и готов к работе!")
            
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при запуске бота: {e}")
            await self.shutdown()

async def main():
    """Основная функция"""
    bot = NewsBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt")
        await bot.shutdown()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        await bot.shutdown()

if __name__ == "__main__":
    os.makedirs('config', exist_ok=True)
    os.makedirs('data/chats', exist_ok=True)
    os.makedirs('session', exist_ok=True)
    
    if not os.path.exists('config/tg_config.json'):
        logger.error("❌ Файл config/tg_config.json не найден!")
        logger.info("📝 Создайте файл с следующими полями:")
        logger.info('''
{
    "bot_token": "YOUR_BOT_TOKEN",
    "api_id": 12345678,
    "api_hash": "YOUR_API_HASH"
}
        ''')
        sys.exit(1)
        
    if not os.path.exists('config/channel_config.json'):
        logger.error("❌ Файл config/channel_config.json не найден!")
        logger.info("📝 Создайте файл с следующими полями:")
        logger.info('''
{
    "channel_id": -1001234567890
}
        ''')
        logger.info("💡 Инструкция по настройке приватного канала:")
        logger.info("1. Создайте ПРИВАТНЫЙ канал в Telegram")
        logger.info("2. Добавьте бота в канал как администратора")
        logger.info("3. Дайте боту права на отправку сообщений")
        logger.info("4. Получите ID канала (отрицательное число)")
        logger.info("5. Укажите ID в channel_config.json")
        sys.exit(1)
    
    asyncio.run(main())