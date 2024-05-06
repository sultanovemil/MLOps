import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessing_std_scal():
    # Получаем данные из файла
    cardio_data_df = pd.read_csv("datasets/cardiovascular_train.csv")
    
    # Создаем список столбцов с количественными признаками
    num_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',]
    
    # Создаем экземпляр StandardScaler
    scaler = StandardScaler()

    # Применяем StandardScaler к количественным столбцам
    cardio_data_df[num_columns] = scaler.fit_transform(cardio_data_df[num_columns])

    # Сохраняем тренировочный датасет в файл
    cardio_data_df.to_csv("datasets/cardiovascular_train.csv", mode="w", index=False)


preprocessing_std_scal()
    