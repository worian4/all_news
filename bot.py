import asyncio
import os
import sys
import signal
from core.news_bot import NewsBot
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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