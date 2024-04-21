# Загрузка необходимых библиотек
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def download_data():
    # Загрузка датасета diabetes с помощью fetch_openml
    cardio_data = fetch_openml("Cardiovascular-Disease-dataset", version=1, parser="auto")
    cardio_data_df = cardio_data.frame
    cardio_data_df['cardio'] = cardio_data_df['cardio'].apply(lambda x: 'positive' if x=='1' else 'negative')
    
    # Разделение данных на обучающую и тестовую выборки
    train_set, test_set = train_test_split(cardio_data_df, test_size=0.1, random_state=42)
    
    return train_set, test_set


