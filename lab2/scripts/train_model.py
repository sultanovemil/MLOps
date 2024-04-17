# Загрузка необходимых библиотек
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


def train_model():
    # Загрузка тренировочных данных из CSV файла 'data_train.csv'
    data_train = pd.read_csv('data_train.csv')
    X_train = data_train[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].values
    y_train = data_train['class'].values

    # Выборка признаков (X_train) и целевой переменной (y_train) из тренировочных данных
    model = LogisticRegression(max_iter=100000).fit(X_train, y_train)
    
    # Сохранение обученной модели в файл 'model.pkl' с помощью pickle
    pickle.dump(model, open('model.pkl', 'wb'))


# if __name__ == "__main__":
train_model()
