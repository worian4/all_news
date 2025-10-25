#!/bin/bash

echo "Установка News Aggregator Bot..."

# Обновление системы
sudo apt update
sudo apt upgrade -y

# Установка Python и pip
sudo apt install -y python3 python3-pip python3-venv

# Создание виртуального окружения
python3 -m venv news_bot_env
source news_bot_env/bin/activate

# Установка зависимостей
pip install -r requirements.txt

echo "Установка завершена!"
echo "Не забудьте:"
echo "1. Заполнить config.json"
echo "2. Получить API_ID и API_HASH на my.telegram.org"
echo "3. Создать бота через @BotFather и получить токен"
echo "4. Запустить бота: python3 news_bot.py"