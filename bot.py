import asyncio
import json
import os
import re
import aiofiles
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError
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
logging.getLogger("telethon").setLevel(logging.INFO)  # Оставляем INFO для Telethon чтобы видеть события

# Настройка GPU для нейросетей
def setup_gpu():
    """Настройка использования GPU для нейросетей"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 Используется GPU: {gpu_name}")
            logger.info(f"🎮 CUDA версия: {torch.version.cuda}")
            logger.info(f"🎮 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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
        return tg_config, constants
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
        return {}, default_constants

TG_CONFIG, CONSTANTS = load_config()

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

    def are_posts_similar(self, fingerprint1, fingerprint2, text1, text2):
        """Проверка схожести двух постов"""
        if fingerprint1 == fingerprint2:
            return True
            
        try:
            # Используем GPU для вычисления схожести
            emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
            emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            return similarity > CONSTANTS['similarity_threshold']
        except Exception as e:
            logger.error(f"Error checking similarity: {e}")
            return False

class ChannelMonitor:
    """Мониторинг каналов через пользовательский аккаунт"""
    
    def __init__(self, api_id, api_hash, neural_processor, bot_application):
        self.api_id = api_id
        self.api_hash = api_hash
        self.neural_processor = neural_processor
        self.bot_application = bot_application
        self.telethon_client = None
        self.is_running = False
        self.user_handlers = {}
        self.monitored_channels = set()
        
    async def start(self):
        """Запуск мониторинга"""
        try:
            from telethon import TelegramClient
            
            logger.info("🔄 Запуск мониторинга каналов...")
            
            self.telethon_client = TelegramClient(
                'user_monitor_session', 
                self.api_id, 
                self.api_hash
            )
            
            await self.telethon_client.start()
            
            me = await self.telethon_client.get_me()
            logger.info(f"✅ Мониторинг запущен от имени: {me.first_name} (@{me.username})")
            
            self.is_running = True
            
            # Тестируем подключение к каналам
            await self._test_channel_connection()
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска мониторинга: {e}")
            raise

    async def _test_channel_connection(self):
        """Тестирование подключения к каналам"""
        try:
            if not self.monitored_channels:
                logger.info("📭 Нет каналов для мониторинга")
                return
                
            logger.info(f"🔍 Тестируем подключение к {len(self.monitored_channels)} каналам...")
            
            for channel in list(self.monitored_channels)[:5]:  # Проверяем первые 5 каналов
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
        
    async def add_user_channels(self, user_id, channels):
        """Добавление каналов для мониторинга пользователя"""
        try:
            from telethon import events
            
            if not self.telethon_client or not self.telethon_client.is_connected():
                await self.start()
            
            logger.info(f"📡 Добавляем каналы для пользователя {user_id}: {channels}")
            
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
            if user_id in self.user_handlers:
                self.telethon_client.remove_event_handler(self.user_handlers[user_id])
                logger.info(f"   🔄 Обновляем обработчик для пользователя {user_id}")
            
            # Создаем новый обработчик для ВСЕХ отслеживаемых каналов
            @self.telethon_client.on(events.NewMessage(chats=list(self.monitored_channels)))
            async def message_handler(event):
                await self._process_new_post(user_id, event.message)
            
            self.user_handlers[user_id] = message_handler
            
            logger.info(f"✅ Добавлены каналы для пользователя {user_id}: {len(new_channels)} новых каналов")
            logger.info(f"📊 Всего отслеживаемых каналов: {len(self.monitored_channels)}")
            
            # Тестируем подключение к новым каналам
            for channel in new_channels:
                try:
                    entity = await self.telethon_client.get_entity(channel)
                    logger.info(f"🔗 Канал подключен: {channel} -> {getattr(entity, 'title', 'Unknown')}")
                except Exception as e:
                    logger.error(f"❌ Ошибка подключения к каналу {channel}: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка добавления каналов для пользователя {user_id}: {e}")
            
    async def remove_user_channels(self, user_id, channels_to_remove):
        """Удаление каналов из мониторинга пользователя"""
        try:
            user_data_path = f"data/users/{user_id}/user_data.json"
            if not os.path.exists(user_data_path):
                return False
                
            user_data = await self._safe_json_load(user_data_path)
            if user_data is None:
                return False
                
            current_channels = user_data.get('channels', [])
            updated_channels = [ch for ch in current_channels if ch not in channels_to_remove]
            
            if len(updated_channels) == len(current_channels):
                return False  # Ничего не изменилось
            
            # Удаляем каналы из общего списка
            for channel in channels_to_remove:
                if channel in self.monitored_channels:
                    self.monitored_channels.remove(channel)
                    logger.info(f"   ➖ Удален канал: {channel}")
            
            user_data['channels'] = updated_channels
            user_data['updated_at'] = datetime.now().isoformat()
            
            await self._safe_json_save(user_data_path, user_data)
            
            # Перезагружаем обработчики с обновленным списком каналов
            if updated_channels:
                await self.add_user_channels(user_id, updated_channels)
            else:
                # Если каналов не осталось, удаляем обработчик
                if user_id in self.user_handlers:
                    self.telethon_client.remove_event_handler(self.user_handlers[user_id])
                    del self.user_handlers[user_id]
                    logger.info(f"   🗑️ Удален обработчик для пользователя {user_id}")
            
            logger.info(f"✅ Удалены каналы для пользователя {user_id}: {len(channels_to_remove)} каналов")
            logger.info(f"📊 Всего отслеживаемых каналов: {len(self.monitored_channels)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления каналов для пользователя {user_id}: {e}")
            return False
            
    async def _process_new_post(self, user_id, message):
        """Обработка нового поста"""
        try:
            logger.info(f"🎯 ПОЛУЧЕНО СООБЩЕНИЕ ИЗ КАНАЛА")
            
            # Пропускаем сообщения без текста (только медиа)
            if not message.text and not message.message:
                logger.info("   📭 Сообщение без текста (только медиа) - пропускаем")
                return
            
            # Получаем текст сообщения
            message_text = message.text or message.message or ""
            logger.info(f"   📝 Текст сообщения: {message_text[:100]}...")
            
            if len(message_text.strip()) < 10:  # Минимальная длина для тестирования
                logger.info(f"   📏 Слишком короткое сообщение ({len(message_text.strip())} chars) - пропускаем")
                return
            
            chat = await message.get_chat()
            channel_username = getattr(chat, 'username', None)
            channel_title = getattr(chat, 'title', 'Unknown Channel')
            
            logger.info(f"   📢 Канал: {channel_title} (@{channel_username})")
            logger.info(f"   🆔 ID сообщения: {message.id}")
            logger.info(f"   📏 Длина текста: {len(message_text)} символов")
            
            # Сохраняем всю информацию о сообщении для пересылки
            post_data = {
                'id': message.id,
                'text': message_text,
                'channel': channel_username if channel_username else channel_title,
                'channel_id': chat.id,
                'message_id': message.id,
                'timestamp': datetime.now().isoformat(),
                'url': f"https://t.me/{channel_username}/{message.id}" if channel_username else f"https://t.me/c/{str(chat.id).replace('-100', '')}/{message.id}",
                'has_media': bool(message.media),
                'is_forward': bool(message.forward)
            }
            
            logger.info("   🧠 Анализируем сообщение нейросетью...")
            
            # Создание отпечатка и оценка интересности
            fingerprint = self.neural_processor.create_fingerprint(post_data['text'])
            interest_score = self.neural_processor.calculate_interest_score(post_data['text'])
            
            post_data['fingerprint'] = fingerprint
            post_data['interest_score'] = interest_score
            
            logger.info(f"   🔑 Отпечаток: {fingerprint[:16]}...")
            logger.info(f"   ⭐ Оценка интересности: {interest_score:.2f}/1.0")
            
            await self._add_to_user_queue(user_id, post_data)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки поста для пользователя {user_id}: {e}")
            
    async def _add_to_user_queue(self, user_id, post_data):
        """Добавление поста в очередь пользователя"""
        try:
            logger.info(f"   📥 Добавляем пост в очередь пользователя {user_id}...")
            
            queue_path = f"data/users/{user_id}/queue.json"
            archive_path = f"data/users/{user_id}/archive.json"
            
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
                if (queued_post.get('fingerprint') == post_data['fingerprint'] or 
                    self.neural_processor.are_posts_similar(
                        queued_post.get('fingerprint'), 
                        post_data['fingerprint'],
                        queued_post.get('text', ''),
                        post_data['text']
                    )):
                    duplicate_index = i
                    logger.info(f"   🔄 Найден дубликат поста в позиции {i}")
                    break
            
            if duplicate_index is not None:
                if post_data['interest_score'] > queue[duplicate_index]['interest_score']:
                    queue[duplicate_index] = post_data
                    logger.info(f"   ✅ Заменен дубликат поста для пользователя {user_id}")
                else:
                    logger.info(f"   📭 Дубликат имеет лучшую оценку, пропускаем")
            else:
                queue.append(post_data)
                logger.info(f"   ✅ Добавлен новый пост в очередь для пользователя {user_id}")
            
            # Безопасное сохранение JSON
            await self._safe_json_save(queue_path, queue)
            logger.info(f"   💾 Очередь сохранена, новый размер: {len(queue)} постов")
                
            await self._update_user_stats(user_id, 'processed')
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в очередь для пользователя {user_id}: {e}")
    
    async def _safe_json_load(self, filepath):
        """Безопасная загрузка JSON файла"""
        try:
            if os.path.exists(filepath):
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():  # Проверяем что файл не пустой
                        return json.loads(content)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка чтения JSON файла {filepath}: {e}")
            # Создаем резервную копию поврежденного файла
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
            
    async def _update_user_stats(self, user_id, stat_type):
        """Обновление статистики пользователя"""
        try:
            user_data_path = f"data/users/{user_id}/user_data.json"
            
            user_data = await self._safe_json_load(user_data_path)
            if user_data is None:
                user_data = {
                    'channels': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_processed': 0,
                    'total_sent': 0
                }
            
            if stat_type == 'processed':
                user_data['total_processed'] = user_data.get('total_processed', 0) + 1
            
            user_data['updated_at'] = datetime.now().isoformat()
            
            await self._safe_json_save(user_data_path, user_data)
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статистики для пользователя {user_id}: {e}")

class NewsBot:
    def __init__(self):
        self.bot_token = TG_CONFIG.get('bot_token', '')
        self.api_id = TG_CONFIG.get('api_id', 0)
        self.api_hash = TG_CONFIG.get('api_hash', '')
        
        if not all([self.bot_token, self.api_id, self.api_hash]):
            logger.error("❌ Не заполнены конфигурационные данные в config/tg_config.json")
            sys.exit(1)
            
        self.application = Application.builder().token(self.bot_token).build()
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
        self.application.add_handler(CommandHandler("stats", self.stats_handler))
        self.application.add_handler(CommandHandler("test_post", self.test_post_handler))
        self.application.add_handler(CommandHandler("monitor_status", self.monitor_status_handler))
        self.application.add_handler(CommandHandler("help", self.help_handler))
        self.application.add_handler(CommandHandler("debug", self.debug_handler))
        
        # Обработчик текстовых сообщений
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
        
        # Обработчик ошибок
        self.application.add_error_handler(self.error_handler)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        try:
            logger.error(f"Exception while handling an update: {context.error}")
            
            # Логируем полную трассировку
            logger.error(f"Traceback: {context.error.__traceback__}")
            
            # Отправляем сообщение пользователю
            if update and update.effective_user:
                await context.bot.send_message(
                    chat_id=update.effective_user.id,
                    text="❌ Произошла ошибка при обработке запроса. Попробуйте еще раз."
                )
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
    
    def get_main_keyboard(self):
        """Клавиатура с основными командами"""
        keyboard = [
            [KeyboardButton("/add_channels"), KeyboardButton("/my_channels")],
            [KeyboardButton("/stats"), KeyboardButton("/remove_channels")],
            [KeyboardButton("/test_post"), KeyboardButton("/monitor_status")],
            [KeyboardButton("/help")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, input_field_placeholder="Выберите команду...")
    
    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_folder = f"data/users/{user_id}"
        os.makedirs(user_folder, exist_ok=True)
        
        await self.create_user_files(user_id)
        
        welcome_text = """
🤖 Добро пожаловать в NewsAggregatorBot!

🎯 **Включено детальное логирование:**
• 📨 Логирование всех постов из каналов
• 🔍 Подробная информация о каждом сообщении
• 🧠 Логи анализа нейросетями
• 📊 Статус добавления в очередь

📋 **Основные команды:**
• /add_channels - добавить каналы
• /my_channels - мои каналы  
• /remove_channels - удалить каналы
• /stats - статистика
• /test_post - тест нейросетей
• /monitor_status - статус мониторинга
• /help - помощь
• /debug - отладочная информация

💡 **Теперь вы увидите в логах все посты из каналов!**
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=self.get_main_keyboard()
        )
    
    async def debug_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отладочная информация"""
        user_id = update.effective_user.id
        
        debug_info = f"""
🔧 **Отладочная информация**

📊 Мониторинг:
• Статус: {'🟢 Активен' if self.channel_monitor.is_running else '🔴 Неактивен'}
• Отслеживаемых каналов: {len(self.channel_monitor.monitored_channels)}
• Обработчиков пользователей: {len(self.channel_monitor.user_handlers)}

📋 Ваши каналы:
"""
        
        user_data = await self.channel_monitor._safe_json_load(f"data/users/{user_id}/user_data.json")
        if user_data and user_data.get('channels'):
            for channel in user_data['channels']:
                debug_info += f"• {channel}\n"
        else:
            debug_info += "• Нет каналов\n"
            
        debug_info += f"""
🎮 Нейросети:
• Устройство: {'🎮 GPU' if str(DEVICE) == 'cuda' else '💻 CPU'}
• Модели загружены: ✅

💡 Проверьте логи в терминале для отслеживания постов!
        """
        
        await update.message.reply_text(
            debug_info,
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

📊 **Информация:**
`/stats` - статистика и метрики
`/monitor_status` - статус мониторинга
`/test_post` - тест работы нейросетей
`/debug` - отладочная информация

❓ **Помощь:**
`/help` - показать это сообщение
`/start` - перезапустить бота

🎯 **Логирование:**
• 📨 Все посты из каналов логируются
• 🔍 Подробная информация о каждом сообщении
• 🧠 Результаты анализа нейросетями
• 📊 Статус добавления в очередь

💡 **Формат каналов:**
t.me/channel_name
@username
https://t.me/channel
        """
        
        await update.message.reply_text(
            help_text,
            reply_markup=self.get_main_keyboard()
        )
    
    async def add_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Присылайте ссылки на Telegram каналы (каждую с новой строки):\n\n"
            "Пример:\n"
            "t.me/rbc_news\n"
            "@meduzaproject\n"
            "https://t.me/rian_ru\n\n"
            "🎯 Теперь включено детальное логирование всех постов!",
            reply_markup=self.get_main_keyboard()
        )

    async def my_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать отслеживаемые каналы"""
        user_id = update.effective_user.id
        
        user_data = await self.channel_monitor._safe_json_load(f"data/users/{user_id}/user_data.json")
        
        if user_data and user_data.get('channels'):
            channels = user_data.get('channels', [])
            channels_text = "\n".join([f"• {channel}" for channel in channels])
            message = f"📋 **Ваши отслеживаемые каналы** ({len(channels)}):\n\n{channels_text}\n\n💡 Используйте /remove_channels чтобы удалить каналы"
        else:
            message = "❌ У вас нет отслеживаемых каналов.\n\n💡 Добавьте каналы командой /add_channels"
        
        await update.message.reply_text(
            message,
            reply_markup=self.get_main_keyboard()
        )

    async def remove_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Удаление каналов из отслеживания"""
        user_id = update.effective_user.id
        
        user_data = await self.channel_monitor._safe_json_load(f"data/users/{user_id}/user_data.json")
        if not user_data:
            await update.message.reply_text(
                "❌ У вас нет отслеживаемых каналов.",
                reply_markup=self.get_main_keyboard()
            )
            return
        
        channels = user_data.get('channels', [])
        
        if not channels:
            await update.message.reply_text(
                "❌ У вас нет отслеживаемых каналов.",
                reply_markup=self.get_main_keyboard()
            )
            return
        
        channels_text = "\n".join([f"• {channel}" for channel in channels])
        
        await update.message.reply_text(
            f"🗑️ **Удаление каналов**\n\n"
            f"Ваши каналы:\n{channels_text}\n\n"
            f"Пришлите каналы для удаления (каждый с новой строки):\n\n"
            f"Пример:\n@channel1\n@channel2",
            reply_markup=self.get_main_keyboard()
        )
        
        # Сохраняем состояние для следующего сообщения
        context.user_data['awaiting_channels_removal'] = True

    async def test_post_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тест обработки поста"""
        test_text = """
        Тестовый пост: В Москве прошло важное совещание по развитию технологий. 
        Эксперты обсудили перспективы искусственного интеллекта и машинного обучения. 
        Были представлены новые исследования в области нейросетей.
        """
        
        fingerprint = self.neural_processor.create_fingerprint(test_text)
        interest_score = self.neural_processor.calculate_interest_score(test_text)
        
        gpu_status = "🎮 (на GPU)" if str(DEVICE) == "cuda" else "💻 (на CPU)"
        
        await update.message.reply_text(
            f"🧪 **Тест обработки поста** {gpu_status}:\n\n"
            f"Текст: {test_text[:100]}...\n"
            f"Отпечаток: {fingerprint[:16]}...\n"
            f"Оценка интересности: {interest_score:.2f}/1.0\n"
            f"⭐ Рейтинг: {'⭐' * int(interest_score * 5)}",
            reply_markup=self.get_main_keyboard()
        )

    async def monitor_status_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статус мониторинга"""
        status = "🟢 Активен" if self.channel_monitor.is_running else "🔴 Неактивен"
        gpu_status = "🎮 GPU" if str(DEVICE) == "cuda" else "💻 CPU"
        
        await update.message.reply_text(
            f"📊 **Статус системы**:\n\n"
            f"Мониторинг: {status}\n"
            f"Пользователей: {len(self.channel_monitor.user_handlers)}\n"
            f"Отслеживаемых каналов: {len(self.channel_monitor.monitored_channels)}\n"
            f"Нейросети: 🟢 Активны ({gpu_status})\n"
            f"Логирование: 📨 Детальное\n"
            f"Пересылка сообщений: 🟢 Включена\n"
            f"Очередь обработки: 🟢 Работает",
            reply_markup=self.get_main_keyboard()
        )

    async def stats_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика пользователя"""
        try:
            user_id = update.effective_user.id
            
            # Безопасная загрузка данных
            user_data = await self.channel_monitor._safe_json_load(f"data/users/{user_id}/user_data.json")
            queue = await self.channel_monitor._safe_json_load(f"data/users/{user_id}/queue.json") or []
            
            gpu_status = "🎮 (GPU)" if str(DEVICE) == "cuda" else "💻 (CPU)"
            
            if user_data:
                stats_text = f"""
📊 **Ваша статистика** {gpu_status}:

• Отслеживаемых каналов: {len(user_data.get('channels', []))}
• Новостей в очереди: {len(queue)}
• Обработано новостей: {user_data.get('total_processed', 0)}
• Отправлено подборок: {user_data.get('total_sent', 0)}
• Логирование: 📨 Детальное

💡 Используйте /my_channels чтобы посмотреть каналы
                """
            else:
                stats_text = "❌ Данные не найдены. Используйте /start для инициализации."
            
            await update.message.reply_text(
                stats_text,
                reply_markup=self.get_main_keyboard()
            )
                
        except Exception as e:
            logger.error(f"Error in stats handler: {e}")
            await update.message.reply_text(
                "❌ Ошибка при получении статистики.",
                reply_markup=self.get_main_keyboard()
            )

    async def create_user_files(self, user_id):
        """Создание файлов пользователя"""
        user_folder = f"data/users/{user_id}"
        base_files = {
            'queue.json': [],
            'archive.json': [],
            'user_data.json': {
                'channels': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'total_processed': 0,
                'total_sent': 0
            }
        }
        
        for filename, content in base_files.items():
            filepath = f"{user_folder}/{filename}"
            if not os.path.exists(filepath):
                await self.channel_monitor._safe_json_save(filepath, content)
    
    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        text = update.message.text
        
        # Проверяем, ожидаем ли мы удаление каналов
        if context.user_data.get('awaiting_channels_removal'):
            context.user_data['awaiting_channels_removal'] = False
            channels_to_remove = self.parse_channels(text)
            
            if channels_to_remove:
                success = await self.channel_monitor.remove_user_channels(user_id, channels_to_remove)
                if success:
                    await update.message.reply_text(
                        f"✅ Удалено {len(channels_to_remove)} каналов:\n" +
                        "\n".join(f"• {ch}" for ch in channels_to_remove),
                        reply_markup=self.get_main_keyboard()
                    )
                else:
                    await update.message.reply_text(
                        "❌ Не удалось удалить каналы. Проверьте правильность ввода.",
                        reply_markup=self.get_main_keyboard()
                    )
            else:
                await update.message.reply_text(
                    "❌ Не удалось распознать каналы для удаления.",
                    reply_markup=self.get_main_keyboard()
                )
            return
        
        # Обычная обработка добавления каналов
        if text.startswith('/'):
            return
            
        channels = self.parse_channels(text)
        
        if channels:
            await self.save_user_channels(user_id, channels)
            await update.message.reply_text(
                f"✅ Добавлено {len(channels)} каналов:\n" +
                "\n".join(f"• {ch}" for ch in channels) +
                f"\n\n🚀 Начинаю мониторинг в реальном времени!\n"
                f"📨 Сообщения будут пересылаться из оригинальных каналов!\n"
                f"📝 Включено детальное логирование всех постов!",
                reply_markup=self.get_main_keyboard()
            )
            
            await self.channel_monitor.add_user_channels(user_id, channels)
            
        else:
            await update.message.reply_text(
                "❌ Не удалось распознать каналы. Используйте форматы:\n"
                "t.me/channel_name\n@channel_name\nhttps://t.me/channel\n\n"
                "Или используйте команду /add_channels",
                reply_markup=self.get_main_keyboard()
            )
    
    def parse_channels(self, text):
        channels = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if 't.me/' in line:
                match = re.search(r't\.me/([a-zA-Z0-9_]+)', line)
                if match:
                    channels.append(f"@{match.group(1)}")
            elif line.startswith('@'):
                channels.append(line)
            elif line.startswith('https://t.me/'):
                match = re.search(r'https://t\.me/([a-zA-Z0-9_]+)', line)
                if match:
                    channels.append(f"@{match.group(1)}")
        
        return list(set(channels))
    
    async def save_user_channels(self, user_id, channels):
        user_data_path = f"data/users/{user_id}/user_data.json"
        
        user_data = await self.channel_monitor._safe_json_load(user_data_path)
        if user_data is None:
            user_data = {
                'channels': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'total_processed': 0,
                'total_sent': 0
            }
        
        existing_channels = set(user_data.get('channels', []))
        new_channels = set(channels)
        all_channels = list(existing_channels.union(new_channels))
        
        user_data['channels'] = all_channels
        user_data['updated_at'] = datetime.now().isoformat()
        
        await self.channel_monitor._safe_json_save(user_data_path, user_data)
        
        logger.info(f"User {user_id} channels updated: {len(all_channels)} channels")
    
    async def process_queue(self):
        """Обработка очереди"""
        while True:
            try:
                if not os.path.exists('data/users'):
                    await asyncio.sleep(CONSTANTS['queue_processing_interval'])
                    continue
                    
                users_folders = os.listdir('data/users')
                for user_folder in users_folders:
                    user_id = user_folder
                    await self.process_user_queue(user_id)
                
                await asyncio.sleep(CONSTANTS['queue_processing_interval'])
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(60)
    
    async def process_user_queue(self, user_id):
        """Обработка очереди пользователя"""
        try:
            queue_path = f"data/users/{user_id}/queue.json"
            queue = await self.channel_monitor._safe_json_load(queue_path) or []
            
            if queue:
                logger.info(f"📊 Обработка очереди пользователя {user_id}: {len(queue)} постов")
            
            archive_path = f"data/users/{user_id}/archive.json"
            archive = await self.channel_monitor._safe_json_load(archive_path) or []
            
            now = datetime.now()
            posts_to_send = []
            updated_queue = []
            
            for post in queue:
                try:
                    post_time = datetime.fromisoformat(post['timestamp'])
                    time_in_queue = now - post_time
                    
                    if time_in_queue.total_seconds() >= CONSTANTS['queue_ttl_seconds']:
                        posts_to_send.append(post)
                        archive.append({
                            'fingerprint': post['fingerprint'],
                            'archived_at': now.isoformat(),
                            'original_channel': post.get('channel'),
                            'interest_score': post.get('interest_score', 0)
                        })
                        logger.info(f"   📤 Готов к отправке: {post.get('channel')} - {post['text'][:50]}...")
                    else:
                        updated_queue.append(post)
                except Exception as e:
                    logger.error(f"Error processing post in queue: {e}")
                    continue
            
            if posts_to_send:
                logger.info(f"   🚀 Отправляем {len(posts_to_send)} постов пользователю {user_id}")
                await self.send_posts_to_user(user_id, posts_to_send)
                await self.update_user_stats(user_id, 'sent')
            
            await self.channel_monitor._safe_json_save(queue_path, updated_queue)
            await self.channel_monitor._safe_json_save(archive_path, archive)
                
            if posts_to_send:
                logger.info(f"✅ Отправлено {len(posts_to_send)} постов пользователю {user_id}")
                
        except Exception as e:
            logger.error(f"Error processing user queue {user_id}: {e}")
    
    async def send_posts_to_user(self, user_id, posts):
        """Отправка постов пользователю с пересылкой оригинальных сообщений"""
        try:
            posts.sort(key=lambda x: x.get('interest_score', 0), reverse=True)
            top_posts = posts[:CONSTANTS['max_posts_per_batch']]
            
            if not top_posts:
                return
            
            logger.info(f"📨 Отправка подборки пользователю {user_id}: {len(top_posts)} постов")
            
            await self.application.bot.send_message(
                chat_id=user_id,
                text=f"📰 Новая подборка новостей ({len(top_posts)} из {len(posts)})",
                reply_markup=self.get_main_keyboard()
            )
            
            for i, post in enumerate(top_posts, 1):
                try:
                    logger.info(f"   📤 Отправка поста {i}/{len(top_posts)}: {post.get('channel')}")
                    
                    # Пытаемся переслать оригинальное сообщение
                    if self.channel_monitor.telethon_client and self.channel_monitor.telethon_client.is_connected():
                        await self.forward_original_message(user_id, post, i)
                    else:
                        # Fallback: отправляем текстовое сообщение
                        await self.send_text_message(user_id, post, i)
                        
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    # Fallback на текстовое сообщение при ошибке
                    await self.send_text_message(user_id, post, i)
            
            logger.info(f"✅ Подборка отправлена пользователю {user_id}")
            
        except Exception as e:
            logger.error(f"Error in send_posts_to_user for {user_id}: {e}")
    
    async def forward_original_message(self, user_id, post, index):
        """Пересылка оригинального сообщения из канала"""
        try:
            from telethon.tl.types import InputPeerChannel
            
            # Получаем информацию о канале
            channel_entity = await self.channel_monitor.telethon_client.get_entity(post['channel_id'])
            
            # Пересылаем сообщение
            await self.channel_monitor.telethon_client.forward_messages(
                entity=user_id,
                messages=post['message_id'],
                from_peer=channel_entity
            )
            
            # Отправляем рейтинг отдельным сообщением
            score = post.get('interest_score', 0.5)
            stars = "⭐" * int(score * 5) + "☆" * (5 - int(score * 5))
            
            await self.application.bot.send_message(
                chat_id=user_id,
                text=f"#{index} Рейтинг: {stars} ({score:.2f}/1.0)",
                reply_to_message_id=None
            )
            
            await asyncio.sleep(1)  # Задержка между сообщениями
            
        except Exception as e:
            logger.error(f"Error forwarding message for user {user_id}: {e}")
            raise
    
    async def send_text_message(self, user_id, post, index):
        """Отправка текстового сообщения (fallback)"""
        text_preview = post['text'][:600] + "..." if len(post['text']) > 600 else post['text']
        score = post.get('interest_score', 0.5)
        stars = "⭐" * int(score * 5) + "☆" * (5 - int(score * 5))
        
        message = f"""
#{index} {post.get('channel', 'Channel')} {stars}

{text_preview}

📖 Читать полностью: {post.get('url', '')}
        """.strip()
        
        await self.application.bot.send_message(
            chat_id=user_id,
            text=message,
            disable_web_page_preview=False
        )
        await asyncio.sleep(1)
    
    async def update_user_stats(self, user_id, stat_type):
        """Обновление статистики"""
        try:
            user_data_path = f"data/users/{user_id}/user_data.json"
            
            user_data = await self.channel_monitor._safe_json_load(user_data_path)
            if user_data is None:
                user_data = {
                    'channels': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_processed': 0,
                    'total_sent': 0
                }
            
            if stat_type == 'sent':
                user_data['total_sent'] = user_data.get('total_sent', 0) + 1
            
            user_data['updated_at'] = datetime.now().isoformat()
            
            await self.channel_monitor._safe_json_save(user_data_path, user_data)
        except Exception as e:
            logger.error(f"Error updating stats for user {user_id}: {e}")
    
    async def cleanup_archive(self):
        """Очистка архива"""
        while True:
            try:
                if not os.path.exists('data/users'):
                    await asyncio.sleep(CONSTANTS['archive_cleanup_interval'])
                    continue
                    
                users_folders = os.listdir('data/users')
                for user_folder in users_folders:
                    user_id = user_folder
                    archive_path = f"data/users/{user_id}/archive.json"
                    
                    archive = await self.channel_monitor._safe_json_load(archive_path) or []
                    
                    now = datetime.now()
                    updated_archive = []
                    
                    for archived_item in archive:
                        try:
                            archived_time = datetime.fromisoformat(archived_item['archived_at'])
                            if (now - archived_time).days < CONSTANTS['archive_ttl_days']:
                                updated_archive.append(archived_item)
                        except Exception as e:
                            logger.error(f"Error processing archive item: {e}")
                            continue
                    
                    if len(updated_archive) != len(archive):
                        await self.channel_monitor._safe_json_save(archive_path, updated_archive)
                
                await asyncio.sleep(CONSTANTS['archive_cleanup_interval'])
            except Exception as e:
                logger.error(f"Error cleaning archive: {e}")
                await asyncio.sleep(3600)
    
    async def shutdown(self):
        """Завершение работы"""
        logger.info("Завершение работы бота...")
        try:
            await self.channel_monitor.stop()
            await self.application.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            logger.info("Бот завершил работу")
    
    async def run(self):
        """Запуск бота"""
        logger.info("🚀 Starting News Aggregator Bot...")
        logger.info("📝 Включено детальное логирование всех постов из каналов!")
        
        try:
            # Запуск фоновых задач
            asyncio.create_task(self.process_queue())
            asyncio.create_task(self.cleanup_archive())
            
            # Запуск бота
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("✅ Бот успешно запущен!")
            
            # Запуск мониторинга
            try:
                await self.channel_monitor.start()
                logger.info("🎯 Мониторинг каналов: АКТИВЕН")
                logger.info("📨 Режим пересылки сообщений: ВКЛЮЧЕН")
                logger.info("🔍 Детальное логирование: ВКЛЮЧЕНО")
            except Exception as e:
                logger.error(f"❌ Мониторинг не запущен: {e}")
                logger.info("💡 Для включения мониторинга нужно авторизоваться через setup_monitor.py")
            
            while True:
                await asyncio.sleep(3600)
                
        except Exception as e:
            logger.error(f"Error running bot: {e}")
        finally:
            await self.shutdown()

def main():
    """Главная функция"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        bot = NewsBot()
        loop.run_until_complete(bot.run())
        
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Программа завершена")

if __name__ == '__main__':
    main()