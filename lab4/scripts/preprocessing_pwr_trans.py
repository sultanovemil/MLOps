import pandas as pd
from sklearn.preprocessing import PowerTransformer


def preprocessing_pwr_trans():
    # Получаем данные из файла
    cardio_data_df = pd.read_csv("datasets/cardiovascular_train.csv")
    
    # Создаем список столбцов с количественными признаками
    num_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',]
    
    # Создаем экземпляр PowerTransformer
    power = PowerTransformer()

    # Применяем PowerTransformer к количественным столбцам
    cardio_data_df[num_columns] = power.fit_transform(cardio_data_df[num_columns])

    # Сохраняем тренировочный датасет в файл
    cardio_data_df.to_csv("datasets/cardiovascular_train.csv", mode="w", index=False)


preprocessing_pwr_trans()