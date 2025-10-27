import asyncio
import json
import os
import re
import aiofiles
import signal
import sys
import logging
from datetime import datetime, timedelta

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import TelegramError, NetworkError

from core.neural_processor import NeuralNewsProcessor
from core.channel_monitor import ChannelMonitor
from core.source.message_texts import *

logger = logging.getLogger(__name__)

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
        self.channel_monitor = ChannelMonitor(self.api_id, self.api_hash, self.neural_processor, self.application, CHANNEL_CONFIG)
        
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

                # ЕСЛИ сообщение из приватного канала-посредника - ИГНОРИРУЕМ ошибку полностью
                if (update.effective_message.chat and 
                    update.effective_message.chat.id == self.channel_monitor.intermediate_channel_id):
                    logger.debug("🔇 Игнорируем ошибку из приватного канала-посредника")
                    return

            logger.error(f"Exception while handling an update: {context.error}")
            
            if isinstance(context.error, NetworkError):
                logger.warning(f"Network error occurred: {context.error}")
                return
            
            logger.error(f"Traceback: {context.error.__traceback__}")
            
            # Отправляем сообщение об ошибке только если это не пересланное сообщение и не приватный канал
            if (update and update.effective_chat and 
                update.effective_chat.id != self.channel_monitor.intermediate_channel_id):
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
            await query.edit_message_reply_markup(reply_markup=keyboard)
        except Exception as e:
            logger.error(f"❌ Ошибка обновления клавиатуры: {e}")
            await query.answer("❌ Ошибка обновления", show_alert=True)

    async def _handle_confirm_remove(self, query, context, chat_id):
        """Подтверждение удаления выбранных каналов"""
        remove_data = context.chat_data.get('remove_channels')
        if not remove_data:
            await query.answer("❌ Сессия устарела. Начните заново.", show_alert=True)
            return
        
        channels = remove_data['available_channels']
        selected_indices = remove_data['selected_indices']
        
        if not selected_indices:
            await query.answer("❌ Не выбрано ни одного канала для удаления", show_alert=True)
            return
        
        # Получаем каналы для удаления
        channels_to_remove = [channels[i] for i in selected_indices]
        
        # Удаляем каналы
        success = await self.channel_monitor.remove_channel_monitoring(chat_id, channels_to_remove)
        
        if success:
            # Форматируем список удаленных каналов
            removed_list = "\n".join([f"• {ch}" for ch in channels_to_remove])
            response_text = SUCCESS_REMOVE_TEXT.format(removed_list=removed_list)
            
            try:
                await query.edit_message_text(
                    response_text,
                    reply_markup=None,
                    parse_mode='Markdown'  # Добавляем parse_mode
                )
            except Exception as e:
                logger.error(f"❌ Ошибка редактирования сообщения: {e}")
                # Пробуем без Markdown
                try:
                    await query.edit_message_text(
                        f"✅ Успешно удалены каналы:\n\n{removed_list}\n\n📊 Используйте /my_channels для просмотра текущего списка.",
                        reply_markup=None
                    )
                except Exception as e2:
                    logger.error(f"❌ Критическая ошибка: {e2}")
        else:
            try:
                await query.edit_message_text(
                    REMOVE_ERROR_TEXT,
                    reply_markup=None,
                    parse_mode='Markdown'  # Добавляем parse_mode
                )
            except Exception as e:
                logger.error(f"❌ Ошибка редактирования сообщения об ошибке: {e}")
                # Пробуем без Markdown
                try:
                    await query.edit_message_text(
                        "❌ Произошла ошибка при удалении каналов. Попробуйте еще раз.",
                        reply_markup=None
                    )
                except Exception as e2:
                    logger.error(f"❌ Критическая ошибка: {e2}")
        
        # Очищаем временные данные
        if 'remove_channels' in context.chat_data:
            del context.chat_data['remove_channels']

    async def _handle_cancel_remove(self, query, context):
        """Отмена удаления каналов"""
        # Очищаем временные данные
        if 'remove_channels' in context.chat_data:
            del context.chat_data['remove_channels']
        
        try:
            await query.edit_message_text(
                REMOVE_CANCELED_TEXT,
                reply_markup=None,
                parse_mode='Markdown'  # Добавляем parse_mode
            )
        except Exception as e:
            logger.error(f"❌ Ошибка при отмене удаления: {e}")
            # Если не получается отредактировать с Markdown, пробуем без него
            try:
                await query.edit_message_text(
                    "❌ Удаление каналов отменено.\n\n💡 Используйте /my_channels для просмотра текущего списка каналов.",
                    reply_markup=None
                )
            except Exception as e2:
                logger.error(f"❌ Критическая ошибка при отмене: {e2}")

    async def _handle_callback_error(self, query, context, chat_id):
        """Обработка ошибок в callback"""
        try:
            await query.edit_message_text(
                CALLBACK_ERROR_TEXT,
                reply_markup=None,
                parse_mode='Markdown'  # Добавляем parse_mode
            )
        except Exception as e:
            logger.error(f"❌ Ошибка при отправке сообщения об ошибке: {e}")
            # Пробуем без Markdown
            try:
                await query.edit_message_text(
                    "❌ Произошла ошибка при обработке запроса. Попробуйте еще раз.",
                    reply_markup=None
                )
            except Exception as e2:
                logger.error(f"❌ Критическая ошибка отправки сообщения: {e2}")
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ Произошла ошибка при обработке запроса. Попробуйте еще раз."
                    )
                except Exception as e3:
                    logger.error(f"❌ Полная потеря связи: {e3}")

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        chat_id = update.effective_chat.id
        user = update.effective_user
        
        logger.info(f"🚀 Команда /start от пользователя {user.first_name} (ID: {chat_id})")
        
        # Создаем папку для данных чата
        os.makedirs(f'data/chats/{chat_id}', exist_ok=True)
        
        # Загружаем или создаем данные чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data:
            chat_data = {
                'channels': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'total_processed': 0,
                'total_sent': 0,
                'chat_type': 'private' if update.effective_chat.type == 'private' else 'group',
                'is_active': True
            }
            await self._safe_json_save(chat_data_path, chat_data)
        
        # Создаем клавиатуру с основными командами
        keyboard = [
            [KeyboardButton("/add_channels"), KeyboardButton("/my_channels")],
            [KeyboardButton("/remove_channels"), KeyboardButton("/help")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"✅ Бот активирован для чата {chat_id}")

    async def add_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /add_channels"""
        chat_id = update.effective_chat.id
        
        logger.info(f"📥 Команда /add_channels от чата {chat_id}")
        
        # Проверяем, что бот активирован для этого чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data:
            await update.message.reply_text(BOT_NOT_ACTIVATED_TEXT)
            return
        
        # Сохраняем состояние ожидания каналов
        context.chat_data['waiting_for_channels'] = True
        
        await update.message.reply_text(
            ADD_CHANNELS_INSTRUCTION,
            parse_mode='Markdown'
        )

    async def my_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /my_channels"""
        chat_id = update.effective_chat.id
        
        logger.info(f"📋 Команда /my_channels от чата {chat_id}")
        
        # Загружаем данные чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data or not chat_data.get('channels'):
            await update.message.reply_text(NO_CHANNELS_TEXT)
            return
        
        channels = chat_data['channels']
        
        # Форматируем список каналов
        channels_list = "\n".join([f"{i}. `{channel}`" for i, channel in enumerate(channels, 1)])
        
        # Форматируем статус
        status = "🟢 Активен" if chat_data.get('is_active', True) else "🔴 Неактивен"
        
        response_text = CHANNELS_LIST_TEXT.format(
            channels_list=channels_list,
            total_processed=chat_data.get('total_processed', 0),
            total_sent=chat_data.get('total_sent', 0),
            channels_count=len(channels),
            status=status
        )
        
        await update.message.reply_text(
            response_text,
            parse_mode='Markdown'
        )

    async def remove_channels_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /remove_channels"""
        chat_id = update.effective_chat.id
        
        logger.info(f"🗑️ Команда /remove_channels от чата {chat_id}")
        
        # Загружаем данные чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data or not chat_data.get('channels'):
            await update.message.reply_text(NO_CHANNELS_TO_REMOVE)
            return
        
        channels = chat_data['channels']
        
        # Сохраняем данные для callback handler
        context.chat_data['remove_channels'] = {
            'available_channels': channels,
            'selected_indices': []
        }
        
        # Создаем интерактивную клавиатуру
        keyboard = self._create_channels_keyboard(channels, [])
        
        await update.message.reply_text(
            REMOVE_CHANNELS_TEXT,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )

    async def stop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stop"""
        chat_id = update.effective_chat.id
        
        logger.info(f"⏹️ Команда /stop от чата {chat_id}")
        
        # Загружаем данные чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data:
            await update.message.reply_text(BOT_NOT_ACTIVATED_TEXT)
            return
        
        # Деактивируем бота для этого чата
        chat_data['is_active'] = False
        chat_data['updated_at'] = datetime.now().isoformat()
        
        await self._safe_json_save(chat_data_path, chat_data)
        
        stop_text = STOP_TEXT.format(
            total_processed=chat_data.get('total_processed', 0),
            total_sent=chat_data.get('total_sent', 0),
            channels_count=len(chat_data.get('channels', []))
        )
        
        await update.message.reply_text(
            stop_text,
            parse_mode='Markdown'
        )
        
        logger.info(f"✅ Бот остановлен для чата {chat_id}")

    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        await update.message.reply_text(
            HELP_TEXT,
            parse_mode='Markdown'
        )

    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        chat_id = update.effective_chat.id
        message_text = update.message.text
        
        logger.info(f"📨 Текстовое сообщение от чата {chat_id}: {message_text[:50]}...")
        
        # Проверяем, ожидаем ли мы каналы от пользователя
        if context.chat_data.get('waiting_for_channels'):
            await self._process_channels_input(update, context, message_text)
            return
        
        # Если это не команда и не ожидаемый ввод, показываем справку
        await update.message.reply_text(
            UNKNOWN_COMMAND_TEXT,
            parse_mode='Markdown'
        )

    async def _process_channels_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: str):
        """Обработка введенных пользователем каналов"""
        chat_id = update.effective_chat.id
        
        # Очищаем состояние ожидания
        context.chat_data['waiting_for_channels'] = False
        
        # Парсим каналы из сообщения
        channels = self._parse_channels_from_text(message_text)
        
        if not channels:
            await update.message.reply_text(INVALID_CHANNELS_TEXT)
            return
        
        logger.info(f"📥 Получены каналы для чата {chat_id}: {channels}")
        
        # Загружаем текущие данные чата
        chat_data_path = f'data/chats/{chat_id}/chat_data.json'
        chat_data = await self._safe_json_load(chat_data_path)
        
        if not chat_data:
            chat_data = {
                'channels': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'total_processed': 0,
                'total_sent': 0,
                'chat_type': 'private',
                'is_active': True
            }
        
        # Добавляем новые каналы (без дубликатов)
        current_channels = set(chat_data.get('channels', []))
        new_channels = [ch for ch in channels if ch not in current_channels]
        duplicate_channels = [ch for ch in channels if ch in current_channels]
        
        if not new_channels:
            await update.message.reply_text(ALL_CHANNELS_EXIST_TEXT)
            return
        
        # Обновляем список каналов
        chat_data['channels'] = list(current_channels.union(set(new_channels)))
        chat_data['updated_at'] = datetime.now().isoformat()
        
        # Сохраняем обновленные данные
        await self._safe_json_save(chat_data_path, chat_data)
        
        # Добавляем каналы в мониторинг
        await self.channel_monitor.add_channel_monitoring(chat_id, new_channels)
        
        # Форматируем ответ
        new_channels_text = "\n".join([f"• `{channel}`" for channel in new_channels]) if new_channels else "—"
        duplicate_channels_text = "\n".join([f"• `{channel}`" for channel in duplicate_channels]) if duplicate_channels else "—"
        
        response_text = SUCCESS_ADD_TEXT.format(
            new_channels=new_channels_text,
            duplicate_channels=duplicate_channels_text,
            total_count=len(chat_data['channels'])
        )
        
        await update.message.reply_text(
            response_text,
            parse_mode='Markdown'
        )
        
        logger.info(f"✅ Добавлено {len(new_channels)} каналов для чата {chat_id}")

    def _parse_channels_from_text(self, text: str):
        """Парсинг каналов из текста сообщения"""
        channels = []
        
        # Разбиваем текст на строки
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Паттерны для распознавания каналов
            patterns = [
                r'@([a-zA-Z0-9_]{5,32})',  # @username
                r't\.me/([a-zA-Z0-9_]{5,32})',  # t.me/username
                r't\.me/c/(\d+)',  # t.me/c/1234567890
                r'https://t\.me/([a-zA-Z0-9_]{5,32})',  # https://t.me/username
                r'https://t\.me/c/(\d+)'  # https://t.me/c/1234567890
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if pattern.startswith('@') or 't.me/' in pattern and not pattern.startswith('t.me/c/'):
                        # username формат
                        channel = f"@{match}"
                    else:
                        # ID канала формат
                        channel = f"https://t.me/c/{match}"
                    
                    if channel not in channels:
                        channels.append(channel)
        
        return channels

    async def _safe_json_load(self, filepath):
        """Безопасная загрузка JSON файла"""
        try:
            if os.path.exists(filepath):
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки JSON {filepath}: {e}")
            return None

    async def _safe_json_save(self, filepath, data):
        """Безопасное сохранение JSON файла"""
        try:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения JSON {filepath}: {e}")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запуск News Bot...")
            
            # Запускаем мониторинг каналов
            await self.channel_monitor.start()
            
            # Восстанавливаем отслеживание каналов для всех чатов
            await self.restore_channel_monitoring()
            
            # Запускаем обработку очереди
            asyncio.create_task(self.process_queue_loop())
            asyncio.create_task(self.cleanup_archive_loop())
            
            # Запускаем бота
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("✅ News Bot успешно запущен!")
            logger.info("🤖 Бот готов к работе")
            logger.info("📡 Мониторинг каналов активен")
            
            # Бесконечный цикл
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при запуске бота: {e}")
            await self.shutdown()

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
                    
                    chat_data = await self._safe_json_load(chat_data_path)
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
            
            queue = await self._safe_json_load(queue_path) or []
            archive = await self._safe_json_load(archive_path) or []
            
            if not queue:
                return
            
            # Проверяем активность бота для этого чата
            chat_data_path = f"data/chats/{chat_id}/chat_data.json"
            chat_data = await self._safe_json_load(chat_data_path)
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
                        # УДАЛЯЕМ ПОСТ ИЗ ОЧЕРЕДИ ПРИ ЛЮБОЙ ОШИБКЕ, ЧТОБЫ НЕ ЗАСОРЯТЬ ОЧЕРЕДЬ
                        # Но оставляем в архиве для отслеживания
                        archive.append({**post, 'error': True, 'error_time': datetime.now().isoformat()})
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка пересылки поста в чат {chat_id}: {e}")
                    # Тоже удаляем пост из очереди при критической ошибке
                    archive.append({**post, 'error': True, 'error_time': datetime.now().isoformat(), 'error_message': str(e)})
            
            # Удаляем обработанные посты из очереди (все, даже те что не удалось отправить)
            remaining_queue = queue[CONSTANTS['max_posts_per_batch']:]
            
            await self._safe_json_save(queue_path, remaining_queue)
            await self._safe_json_save(archive_path, archive)
            
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
            try:
                original_message = await self.channel_monitor.telethon_client.get_messages(
                    post_data['original_channel_id'],
                    ids=post_data['original_message_id']
                )
            except Exception as e:
                logger.error(f"❌ Ошибка получения сообщения {post_data['original_message_id']} из канала {post_data['original_channel_id']}: {e}")
                return False
            
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
            intermediate_message_id = None
            if hasattr(forwarded_message, 'id'):
                intermediate_message_id = forwarded_message.id
            elif isinstance(forwarded_message, list) and len(forwarded_message) > 0:
                intermediate_message_id = forwarded_message[0].id
            
            if not intermediate_message_id:
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
                # УДАЛЯЕМ ОТПРАВКУ СООБЩЕНИЯ ОБ ОШИБКЕ В ПРИВАТНЫЙ КАНАЛ
                # Вместо этого просто логируем ошибку
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
            
            chat_data = await self._safe_json_load(chat_data_path)
            if chat_data is None:
                return
            
            chat_data['total_sent'] = chat_data.get('total_sent', 0) + sent_count
            chat_data['updated_at'] = datetime.now().isoformat()
            
            await self._safe_json_save(chat_data_path, chat_data)
            
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
            archive = await self._safe_json_load(archive_path) or []
            
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
                await self._safe_json_save(archive_path, cleaned_archive)
                logger.info(f"🧹 Очищен архив чата {chat_id}: удалено {removed_count} старых постов")
                
        except Exception as e:
            logger.error(f"❌ Ошибка очистки архива чата {chat_id}: {e}")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Завершение работы бота...")
        
        await self.channel_monitor.stop()
        
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
        
        logger.info("👋 Бот завершил работу")