import pickle
import pandas as pd
from sklearn.metrics import f1_score


def test_model():
    # Загрузка сохраненной модели из файла 'model.pkl'
    loaded_model = pickle.load(open('model.pkl', 'rb'))

    # Загрузка тестовых данных из файла 'data_test.csv'
    path = "data_test.csv"
    data = pd.read_csv(path)
    X = data[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].values
    y = data['class'].values

    
   # Определение функции для вычисления метрики F1-Scor
    def calculate_metric(model, X, y):   
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, pos_label='tested_positive')            
        return f1
        

    # Вывод значения среднеквадратичной ошибки (MSE) на экран
    print("F1-Score (positive class):", calculate_metric(loaded_model, X, y))


# if __name__ == "__main__":
test_model()