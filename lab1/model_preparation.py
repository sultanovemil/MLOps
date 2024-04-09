#!/usr/bin/python3

from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

# Путь к файлу с обучающими данными
path = "train/train_scaled_data.csv"

# Загрузка данных в формате DataFrame
data = pd.read_csv(path)

# Инициализация модели линейной регрессии
model = LinearRegression()

# Определение признаков и целевой переменной
X, y = data.drop("target", axis=1), data["target"]

# Обучение модели на данных
model.fit(X, y)

# Сохранение обученной модели в файл 'model.pkl'
pickle.dump(model, open('model.pkl', 'wb'))