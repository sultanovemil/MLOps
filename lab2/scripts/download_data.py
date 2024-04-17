# Загрузка необходимых библиотек
from sklearn.datasets import fetch_openml


def download_data():
    # Загрузка датасета diabetes с помощью fetch_openml
    diabetes = fetch_openml("diabetes", version=1, parser="auto")
    diabetes_df = diabetes.frame

    # Разделение данных на обучающую и тестовую выборки
    train = diabetes_df[:400]  # Обучающая выборка с 400 записями
    test = diabetes_df[401:]  # Тестовая выборка с оставшимися записями

    # Экспорт данных в CSV файлы для дальнейшего использования
    train.to_csv('data_train.csv', index=False)  # Экспорт обучающей выборки в файл data_train.csv
    test.to_csv('data_test.csv', index=False)  # Экспорт тестовой выборки в файл data_test.csv


# if __name__ == "__main__":
download_data()
