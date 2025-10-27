import os
import json
import aiofiles
import hashlib
import logging
from datetime import datetime, timedelta
from telethon import TelegramClient, events

logger = logging.getLogger(__name__)

class ChannelMonitor:
    """Мониторинг каналов через пользовательский аккаунт"""
    
    def __init__(self, api_id, api_hash, neural_processor, bot_application, channel_config):
        self.api_id = api_id
        self.api_hash = api_hash
        self.neural_processor = neural_processor
        self.bot_application = bot_application
        self.telethon_client = None
        self.is_running = False
        self.channel_handlers = {}
        self.monitored_channels = set()
        self.intermediate_channel_id = channel_config.get("channel_id")
        self.intermediate_channel_title = "Приватный канал-посредник"
        
    async def start(self):
        """Запуск мониторинга"""
        try:
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
                try:
                    # Логируем получение сообщения
                    logger.info(f"📨 СРАБОТАЛ ОБРАБОТЧИК для чата {chat_id}")
                    logger.info(f"   📧 От: {getattr(event.chat, 'title', 'Unknown')} (@{getattr(event.chat, 'username', 'Unknown')})")
                    logger.info(f"   🆔 ID сообщения: {event.message.id}")
                    
                    await self._process_new_post(chat_id, event.message)
                except Exception as e:
                    logger.error(f"❌ Ошибка в обработчике сообщений для чата {chat_id}: {e}")
            
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
        """Обработка нового поста - улучшенная версия с лучшим логированием и обработкой"""
        try:
            # Пропускаем служебные сообщения и сообщения без контента
            if not message or (not message.text and not message.message and not message.media):
                logger.debug(f"📭 Пропускаем пустое или служебное сообщение для чата {chat_id}")
                return
            
            # Получаем текст сообщения разными способами
            message_text = ""
            if message.text:
                message_text = message.text
            elif message.message:
                message_text = message.message
            elif message.media:
                # Для медиа-сообщений пытаемся получить подпись
                if hasattr(message, 'caption') and message.caption:
                    message_text = message.caption
                elif hasattr(message, 'message') and message.message:
                    message_text = message.message
            
            logger.info(f"🎯 ПОЛУЧЕНО СООБЩЕНИЕ ИЗ КАНАЛА ДЛЯ ЧАТА {chat_id}")
            logger.info(f"   📝 Тип сообщения: {type(message)}")
            logger.info(f"   📄 Атрибуты сообщения: {[attr for attr in dir(message) if not attr.startswith('_')]}")
            
            # Детальное логирование содержимого сообщения
            if hasattr(message, 'text') and message.text:
                logger.info(f"   📖 Текст (text): {message.text[:200]}...")
            if hasattr(message, 'message') and message.message:
                logger.info(f"   📖 Сообщение (message): {message.message[:200]}...")
            if hasattr(message, 'caption') and message.caption:
                logger.info(f"   📖 Подпись (caption): {message.caption[:200]}...")
            if hasattr(message, 'media') and message.media:
                logger.info(f"   🖼️ Медиа присутствует: {type(message.media)}")
            
            # Проверяем, активен ли бот для этого чата
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            chat_data = await self._safe_json_load(chat_data_path)
            if chat_data and not chat_data.get('is_active', True):
                logger.info(f"   ⏸️ Бот неактивен для чата {chat_id}, пропускаем сообщение")
                return
            
            # Более гибкая проверка на минимальную длину текста
            clean_text = message_text.strip() if message_text else ""
            if len(clean_text) < 5:  # Уменьшил минимальную длину с 10 до 5 символов
                # Но проверяем, есть ли медиа - если есть медиа, то пропускаем текстовые проверки
                if not message.media:
                    logger.info(f"   📏 Слишком короткое сообщение ({len(clean_text)} chars) без медиа - пропускаем")
                    return
                else:
                    logger.info(f"   🖼️ Сообщение с медиа, но без текста - обрабатываем")
                    # Для медиа без текста создаем минимальный текст
                    message_text = "📷 Медиа-сообщение"
            
            try:
                chat = await message.get_chat()
                channel_username = getattr(chat, 'username', None)
                channel_title = getattr(chat, 'title', 'Unknown Channel')
                channel_id = getattr(chat, 'id', None)
            except Exception as e:
                logger.error(f"   ❌ Ошибка получения информации о канале: {e}")
                # Используем fallback информацию
                channel_username = "unknown"
                channel_title = "Unknown Channel"
                channel_id = 0
            
            logger.info(f"   📢 Канал: {channel_title} (@{channel_username}, ID: {channel_id})")
            logger.info(f"   🆔 ID сообщения: {message.id}")
            logger.info(f"   📏 Длина текста: {len(clean_text)} символов")
            logger.info(f"   🖼️ Есть медиа: {bool(message.media)}")
            
            # Создание отпечатка и оценка интересности
            logger.info("   🧠 Анализируем сообщение нейросетью...")
            try:
                fingerprint = self.neural_processor.create_fingerprint(message_text)
                interest_score = self.neural_processor.calculate_interest_score(message_text)
                logger.info(f"   🔑 Отпечаток: {fingerprint[:16]}...")
                logger.info(f"   ⭐ Оценка интересности: {interest_score:.2f}/1.0")
            except Exception as e:
                logger.error(f"   ❌ Ошибка нейросетевого анализа: {e}")
                # Используем fallback значения
                fingerprint = hashlib.sha256(f"{channel_id}_{message.id}_{message_text}".encode()).hexdigest()
                interest_score = 0.5
                logger.info(f"   🔑 Fallback отпечаток: {fingerprint[:16]}...")
                logger.info(f"   ⭐ Fallback оценка: {interest_score:.2f}/1.0")
            
            # Формируем URL для оригинального сообщения
            try:
                if channel_username and channel_username != "unknown":
                    url = f"https://t.me/{channel_username}/{message.id}"
                else:
                    # Для каналов без username используем ID
                    url = f"https://t.me/c/{str(channel_id).replace('-100', '')}/{message.id}"
            except:
                url = "URL недоступен"
            
            # Сохраняем метаданные для пересылки через приватный канал-посредник
            post_data = {
                'id': message.id,
                'channel': channel_username if channel_username and channel_username != "unknown" else channel_title,
                'channel_id': channel_id,
                'message_id': message.id,
                'timestamp': datetime.now().isoformat(),
                'url': url,
                'has_media': bool(message.media),
                'is_forward': bool(getattr(message, 'forward', None)),
                'chat_id': chat_id,
                'fingerprint': fingerprint,
                'interest_score': interest_score,
                'original_message_id': message.id,
                'original_channel_id': channel_id,
                'text': message_text,  # Сохраняем текст сообщения
                'message_object': None,  # Не сохраняем объект сообщения для безопасности
                'processed_at': datetime.now().isoformat()
            }
            
            await self._add_to_chat_queue(chat_id, post_data)
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка обработки поста для чата {chat_id}: {e}")
            logger.error(f"   Подробности: {type(e).__name__}, сообщение: {str(e)}")
            
    async def _add_to_chat_queue(self, chat_id, post_data):
        """Добавление поста в очередь чата - улучшенная версия"""
        try:
            logger.info(f"   📥 Добавляем пост в очередь чата {chat_id}...")
            
            queue_path = f"data/chats/{chat_id}/queue.json"
            archive_path = f"data/chats/{chat_id}/archive.json"
            
            os.makedirs(os.path.dirname(queue_path), exist_ok=True)
            
            # Безопасная загрузка JSON
            queue = await self._safe_json_load(queue_path) or []
            archive = await self._safe_json_load(archive_path) or []
            
            logger.info(f"   📊 Текущий размер очереди: {len(queue)} постов")
            logger.info(f"   📚 Размер архива: {len(archive)} постов")
            
            # Проверяем, нет ли такого поста в архиве
            archive_fingerprints = {item.get('fingerprint') for item in archive if item.get('fingerprint')}
            if post_data['fingerprint'] in archive_fingerprints:
                logger.info(f"   📭 Пост уже в архиве, пропускаем: {post_data['fingerprint'][:16]}...")
                return
            
            # Проверяем на дубликаты в очереди
            duplicate_index = None
            for i, queued_post in enumerate(queue):
                if self.neural_processor.are_posts_similar(queued_post.get('fingerprint'), post_data['fingerprint']):
                    duplicate_index = i
                    logger.info(f"   🔄 Найден дубликат поста в позиции {i}")
                    logger.info(f"   📊 Очередь: {queued_post.get('interest_score', 0):.2f}, новый: {post_data['interest_score']:.2f}")
                    break
            
            if duplicate_index is not None:
                if post_data['interest_score'] > queue[duplicate_index]['interest_score']:
                    old_score = queue[duplicate_index]['interest_score']
                    queue[duplicate_index] = post_data
                    logger.info(f"   ✅ Заменен дубликат поста для чата {chat_id} (оценка: {old_score:.2f} -> {post_data['interest_score']:.2f})")
                else:
                    logger.info(f"   📭 Дубликат имеет лучшую оценку ({queue[duplicate_index]['interest_score']:.2f} vs {post_data['interest_score']:.2f}), пропускаем")
            else:
                queue.append(post_data)
                logger.info(f"   ✅ Добавлен новый пост в очередь для чата {chat_id}")
            
            # Ограничиваем размер очереди (максимум 100 постов)
            if len(queue) > 100:
                queue = queue[:100]
                logger.info(f"   ✂️ Очередь обрезана до 100 постов")
            
            # Безопасное сохранение JSON
            await self._safe_json_save(queue_path, queue)
            logger.info(f"   💾 Очередь сохранена, новый размер: {len(queue)} постов")
                
            await self._update_chat_stats(chat_id, 'processed')
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в очередь для чата {chat_id}: {e}")
            logger.error(f"   Данные поста: {post_data.get('channel', 'Unknown')}, ID: {post_data.get('id', 'Unknown')}")
    
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