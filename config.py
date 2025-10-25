import asyncio
import json
import os
import hashlib
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from telethon import TelegramClient, events
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
import logging
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAggregatorBot:
    def __init__(self, api_id: int, api_hash: str, bot_token: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        self.client = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Базовая директория для данных
        self.base_data_dir = "data"
        os.makedirs(self.base_data_dir, exist_ok=True)
        
        # Загрузка конфигурации
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Загрузка конфигурации из JSON файла"""
        config_path = "config.json"
        default_config = {
            "queue_timeout_minutes": 30,
            "archive_cleanup_days": 7,
            "update_interval_minutes": 5,
            "similarity_threshold": 0.8,
            "min_interest_score": 0.3
        }
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        return default_config
    
    def get_user_dir(self, user_id: int) -> str:
        """Получение пути к директории пользователя"""
        user_dir = os.path.join(self.base_data_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    async def load_user_data(self, user_id: int) -> Dict:
        """Загрузка данных пользователя"""
        user_dir = self.get_user_dir(user_id)
        user_data_path = os.path.join(user_dir, "user_data.json")
        
        default_data = {
            "user_id": user_id,
            "channels": [],
            "username": "",
            "registration_date": datetime.now().isoformat(),
            "settings": {}
        }
        
        try:
            async with aiofiles.open(user_data_path, 'r') as f:
                content = await f.read()
                user_data = json.loads(content)
                default_data.update(user_data)
        except FileNotFoundError:
            user_data = default_data
            await self.save_user_data(user_id, user_data)
        
        return user_data
    
    async def save_user_data(self, user_id: int, user_data: Dict):
        """Сохранение данных пользователя"""
        user_dir = self.get_user_dir(user_id)
        user_data_path = os.path.join(user_dir, "user_data.json")
        
        async with aiofiles.open(user_data_path, 'w') as f:
            await f.write(json.dumps(user_data, indent=4, ensure_ascii=False))
    
    async def load_queue(self, user_id: int) -> List[Dict]:
        """Загрузка очереди пользователя"""
        user_dir = self.get_user_dir(user_id)
        queue_path = os.path.join(user_dir, "queue.json")
        
        try:
            async with aiofiles.open(queue_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except FileNotFoundError:
            return []
    
    async def save_queue(self, user_id: int, queue: List[Dict]):
        """Сохранение очереди пользователя"""
        user_dir = self.get_user_dir(user_id)
        queue_path = os.path.join(user_dir, "queue.json")
        
        async with aiofiles.open(queue_path, 'w') as f:
            await f.write(json.dumps(queue, indent=4, ensure_ascii=False))
    
    async def load_archive(self, user_id: int) -> List[Dict]:
        """Загрузка архива пользователя"""
        user_dir = self.get_user_dir(user_id)
        archive_path = os.path.join(user_dir, "archive.json")
        
        try:
            async with aiofiles.open(archive_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except FileNotFoundError:
            return []
    
    async def save_archive(self, user_id: int, archive: List[Dict]):
        """Сохранение архива пользователя"""
        user_dir = self.get_user_dir(user_id)
        archive_path = os.path.join(user_dir, "archive.json")
        
        async with aiofiles.open(archive_path, 'w') as f:
            await f.write(json.dumps(archive, indent=4, ensure_ascii=False))
    
    def calculate_fingerprint(self, text: str) -> np.ndarray:
        """Расчет отпечатка текста с помощью нейросети"""
        return self.model.encode([text])[0]
    
    def calculate_interest_score(self, message: Message) -> float:
        """Расчет показателя интересности поста"""
        score = 0.0
        
        # Базовый счет за длину текста
        if message.text:
            text_length = len(message.text)
            # Нормализуем длину текста (0-1)
            text_score = min(text_length / 1000, 1.0)
            score += text_score * 0.6
        
        # Бонус за медиа
        if message.media:
            if isinstance(message.media, MessageMediaPhoto):
                score += 0.2
            elif isinstance(message.media, MessageMediaDocument):
                score += 0.15
        
        # Бонус за просмотры и реакции
        if hasattr(message, 'views') and message.views:
            views_score = min(message.views / 1000, 0.2)
            score += views_score
        
        return min(score, 1.0)
    
    def is_similar(self, fingerprint1: List[float], fingerprint2: List[float]) -> bool:
        """Проверка схожести двух отпечатков"""
        similarity = cosine_similarity([fingerprint1], [fingerprint2])[0][0]
        return similarity > self.config["similarity_threshold"]
    
    async def process_message(self, user_id: int, message: Message, channel: str):
        """Обработка нового сообщения"""
        if not message.text or len(message.text.strip()) < 10:
            return
        
        user_data = await self.load_user_data(user_id)
        queue = await self.load_queue(user_id)
        archive = await self.load_archive(user_id)
        
        # Создаем отпечаток
        fingerprint = self.calculate_fingerprint(message.text).tolist()
        interest_score = self.calculate_interest_score(message)
        
        # Проверяем минимальный порог интересности
        if interest_score < self.config["min_interest_score"]:
            return
        
        post_data = {
            "id": f"{channel}_{message.id}",
            "channel": channel,
            "text": message.text,
            "fingerprint": fingerprint,
            "interest_score": interest_score,
            "timestamp": datetime.now().isoformat(),
            "message_id": message.id,
            "media": bool(message.media)
        }
        
        # Проверка на дубликаты в архиве
        for archived_post in archive:
            if self.is_similar(fingerprint, archived_post["fingerprint"]):
                logger.info(f"Пост из архива найден для пользователя {user_id}")
                return
        
        # Проверка на дубликаты в очереди
        duplicate_index = -1
        for i, queued_post in enumerate(queue):
            if self.is_similar(fingerprint, queued_post["fingerprint"]):
                if interest_score > queued_post["interest_score"]:
                    duplicate_index = i
                    logger.info(f"Найден более интересный дубликат для пользователя {user_id}")
                else:
                    logger.info(f"Найден менее интересный дубликат для пользователя {user_id}")
                    return
                break
        
        # Добавляем или заменяем пост в очереди
        if duplicate_index >= 0:
            queue[duplicate_index] = post_data
        else:
            queue.append(post_data)
        
        await self.save_queue(user_id, queue)
        logger.info(f"Пост добавлен в очередь для пользователя {user_id}")
    
    async def send_queued_posts(self, user_id: int):
        """Отправка постов из очереди пользователю"""
        queue = await self.load_queue(user_id)
        archive = await self.load_archive(user_id)
        
        now = datetime.now()
        posts_to_send = []
        updated_queue = []
        
        for post in queue:
            post_time = datetime.fromisoformat(post["timestamp"])
            time_in_queue = now - post_time
            
            if time_in_queue.total_seconds() >= self.config["queue_timeout_minutes"] * 60:
                posts_to_send.append(post)
                archive.append(post)
            else:
                updated_queue.append(post)
        
        # Отправляем посты пользователю
        for post in posts_to_send:
            try:
                message_text = f"📰 **Из канала {post['channel']}:**\n\n{post['text']}\n\n💫 Оценка: {post['interest_score']:.2f}"
                await self.client.send_message(user_id, message_text)
                logger.info(f"Отправлен пост пользователю {user_id}")
            except Exception as e:
                logger.error(f"Ошибка отправки пользователю {user_id}: {e}")
        
        # Обновляем очередь и архив
        await self.save_queue(user_id, updated_queue)
        await self.save_archive(user_id, archive)
        
        # Очистка архива (раз в неделю)
        await self.cleanup_archive(user_id)
    
    async def cleanup_archive(self, user_id: int):
        """Очистка старого архива"""
        archive = await self.load_archive(user_id)
        now = datetime.now()
        cleanup_days = timedelta(days=self.config["archive_cleanup_days"])
        
        cleaned_archive = []
        for post in archive:
            post_time = datetime.fromisoformat(post["timestamp"])
            if now - post_time <= cleanup_days:
                cleaned_archive.append(post)
        
        await self.save_archive(user_id, cleaned_archive)
    
    async def parse_channels(self, user_id: int):
        """Парсинг каналов пользователя"""
        user_data = await self.load_user_data(user_id)
        
        for channel in user_data.get("channels", []):
            try:
                async for message in self.client.iter_messages(channel, limit=20):
                    await self.process_message(user_id, message, channel)
            except Exception as e:
                logger.error(f"Ошибка парсинга канала {channel} для пользователя {user_id}: {e}")
    
    async def start_bot(self):
        """Запуск бота"""
        self.client = TelegramClient('news_bot_session', self.api_id, self.api_hash)
        await self.client.start(bot_token=self.bot_token)
        
        logger.info("Бот запущен!")
        
        @self.client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            user_id = event.sender_id
            user_data = await self.load_user_data(user_id)
            
            welcome_text = """
🤖 **Добро пожаловать в News Aggregator Bot!**

Я помогу вам отслеживать новости с разных Telegram каналов и присылать только самые интересные и уникальные посты.

📋 **Доступные команды:**
/add_channels - Добавить каналы для отслеживания
/list_channels - Показать текущие каналы
/remove_channels - Удалить каналы
/stats - Статистика

Чтобы начать, добавьте каналы командой /add_channels
            """
            
            await event.reply(welcome_text)
            logger.info(f"Новый пользователь: {user_id}")
        
        @self.client.on(events.NewMessage(pattern='/add_channels'))
        async def add_channels_handler(event):
            user_id = event.sender_id
            
            instruction_text = """
📥 **Добавление каналов**

Пришлите мне ссылки на каналы через запятую или каждую с новой строки.

Пример:
t.me/rbc_news
@rian_ru
https://t.me/meduzalive

⚠️ **Важно:** Бот должен быть добавлен в канал как администратор (для приватных каналов)
            """
            
            await event.reply(instruction_text)
            
            @self.client.on(events.NewMessage(from_users=user_id))
            async def channels_input_handler(inner_event):
                channels_text = inner_event.text.strip()
                
                if channels_text.startswith('/'):
                    return
                
                channels = []
                for line in channels_text.split('\n'):
                    for channel in line.split(','):
                        channel = channel.strip()
                        if channel:
                            # Нормализация имени канала
                            if 't.me/' in channel:
                                channel = '@' + channel.split('t.me/')[-1]
                            elif not channel.startswith('@'):
                                channel = '@' + channel
                            channels.append(channel)
                
                user_data = await self.load_user_data(user_id)
                user_data["channels"] = list(set(user_data.get("channels", []) + channels))
                await self.save_user_data(user_id, user_data)
                
                await inner_event.reply(f"✅ Добавлено {len(channels)} каналов!\n\nСписок каналов: {', '.join(channels)}")
                
                # Удаляем обработчик чтобы избежать повторного срабатывания
                inner_event.client.remove_event_handler(channels_input_handler)
        
        @self.client.on(events.NewMessage(pattern='/list_channels'))
        async def list_channels_handler(event):
            user_id = event.sender_id
            user_data = await self.load_user_data(user_id)
            channels = user_data.get("channels", [])
            
            if channels:
                await event.reply(f"📋 **Ваши каналы:**\n\n" + "\n".join(channels))
            else:
                await event.reply("❌ Каналы не добавлены. Используйте /add_channels")
        
        @self.client.on(events.NewMessage(pattern='/stats'))
        async def stats_handler(event):
            user_id = event.sender_id
            user_data = await self.load_user_data(user_id)
            queue = await self.load_queue(user_id)
            archive = await self.load_archive(user_id)
            
            stats_text = f"""
📊 **Статистика:**

👤 Пользователь: {user_id}
📅 Зарегистрирован: {user_data.get('registration_date', 'Неизвестно')}
📰 Отслеживаемых каналов: {len(user_data.get('channels', []))}
⏳ Постов в очереди: {len(queue)}
📁 Постов в архиве: {len(archive)}
            """
            
            await event.reply(stats_text)
        
        # Запуск фоновых задач
        async def background_tasks():
            while True:
                try:
                    # Получаем всех пользователей
                    for user_dir in os.listdir(self.base_data_dir):
                        if user_dir.isdigit():
                            user_id = int(user_dir)
                            user_data = await self.load_user_data(user_id)
                            
                            if user_data.get("channels"):
                                # Парсинг каналов
                                await self.parse_channels(user_id)
                                # Отправка постов из очереди
                                await self.send_queued_posts(user_id)
                    
                    await asyncio.sleep(self.config["update_interval_minutes"] * 60)
                    
                except Exception as e:
                    logger.error(f"Ошибка в фоновых задачах: {e}")
                    await asyncio.sleep(60)
        
        # Запускаем фоновые задачи
        asyncio.create_task(background_tasks())
        
        # Запускаем клиент
        await self.client.run_until_disconnected()

async def main():
    # Конфигурация (замените на свои данные)
    with open('tg_config.json') as f: d = json.load(f)

    API_ID = d["API_ID"]  # Ваш API ID из my.telegram.org
    API_HASH = d["API_HASH"]  # Ваш API Hash
    BOT_TOKEN = d["BOT_TOKEN"]  # Токен бота от @BotFather
    
    bot = NewsAggregatorBot(API_ID, API_HASH, BOT_TOKEN)
    await bot.start_bot()

if __name__ == "__main__":
    asyncio.run(main())