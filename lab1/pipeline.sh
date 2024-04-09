#!/bin/bash

# Установка Python 3.9 с помощью apt
sudo apt update
sudo apt install python3.9


# Создание виртуального окружения с именем venv
python3.9 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Установка всех необходимых библиотек из файла requirements.txt
pip install -r requirements.txt

# Запуск скрипта data_creation.py
python data_creation.py

# Запуск скрипта model_preprocessing.py
python model_preprocessing.py

# Запуск скрипта model_preparation.py
python model_preparation.py

# Запуск скрипта model_testing.py
python model_testing.py
