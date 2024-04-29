# Загрузка необходимых библиотек
from sklearn.datasets import fetch_openml


def download_data():
    # Получаем набор данных о сердечно-сосудистых заболеваниях из Openml
    cardio_data = fetch_openml("Cardiovascular-Disease-dataset", version=1, parser="auto")
    cardio_data_df = cardio_data.frame

    # Сохранаяем модель в файл
    cardio_data_df.to_csv("datasets/cardiovascular_train.csv", mode="w", index=False)


download_data()