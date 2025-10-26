#!/usr/bin/env python3
"""
Скрипт для настройки мониторинга
"""

import asyncio
import json
from telethon import TelegramClient
import os

async def setup_monitor():
    """Настройка пользовательского аккаунта для мониторинга"""
    
    with open('config/tg_config.json', 'r') as f:
        config = json.load(f)
    
    api_id = config['api_id']
    api_hash = config['api_hash']
    
    print("🔧 Настройка мониторинга каналов")
    print("=" * 50)
    print("Этот шаг необходим для мониторинга Telegram каналов.")
    print("Вам нужно будет войти в ваш Telegram аккаунт.")
    print("=" * 50)

    os.makedirs('session', exist_ok=True)
    client = TelegramClient('session/user_monitor_session', api_id, api_hash)
    
    try:
        await client.start()
        me = await client.get_me()
        print(f"✅ Успешная авторизация!")
        print(f"👤 Имя: {me.first_name}")
        print(f"📱 Username: @{me.username}")
        print(f"🆔 ID: {me.id}")
        print("\n✅ Настройка завершена! Теперь можно запускать бота.")
        
    except Exception as e:
        print(f"❌ Ошибка настройки: {e}")
    finally:
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(setup_monitor())