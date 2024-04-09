#!/usr/bin/python3

import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

# Загрузка сохраненной модели из файла 'model.pkl'
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Загрузка тестовых данных из файла 'test/test_scaled_data.csv' с помощью pandas
path = "test/test_scaled_data.csv"
data = pd.read_csv(path)
X, y = data.drop("target", axis=1), data["target"]

# Определение функции для вычисления метрики (среднеквадратичная ошибка)
def calculate_metric(model, X, y):    
    y_pred = model.predict(X)  # Получение предсказаний модели на входных данных X
    mse = mean_squared_error(y, y_pred)  # Вычисление среднеквадратичной ошибки между реальными и предсказанными значениями    
    return mse

# Вывод значения среднеквадратичной ошибки (MSE) на экран
print("Mean Squared Error (MSE):", calculate_metric(loaded_model, X, y))

