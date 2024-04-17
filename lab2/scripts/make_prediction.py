# Загрузка необходимых библиотек
import pickle
import pandas as pd


def make_prediction():
    # Загрузка ранее сохраненной модели из файла 'model.pkl' с помощью pickle
    loaded_model = pickle.load(open('model.pkl', 'rb'))

    # Загрузка тестовых данных из CSV файла 'data_test.csv'
    data_test = pd.read_csv('data_test.csv')

    # Выборка признаков для тестирования из тестовых данных
    X_test = data_test[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].values

    # Прогнозирование классов для первых 5 наблюдений тестовых данных с помощью загруженной модели и вывод результата
    print(loaded_model.predict(X_test[0:5]))


# if __name__ == "__main__":
make_prediction()
