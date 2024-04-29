from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def preprocessing_one_hot():
    # Получаем данные из файла
    cardio_data_df = pd.read_csv("datasets/cardiovascular_train.csv")
    
    # Создаем список столбцов с категориальными и количественными признаками
    cat_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    num_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',]

    # Создаем экземпляр OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Применяем OneHotEncoder к категориальным столбцам
    data_encoded = pd.DataFrame(encoder.fit_transform(cardio_data_df[cat_columns]).toarray())
    data_encoded.columns = encoder.get_feature_names_out(cat_columns)

    # Объединяем закодированные данные и числовые столбцы
    cardio_data = pd.concat([cardio_data_df[num_columns], data_encoded], axis=1)

    
    # Сохраняем тренировочный датасет в файл
    cardio_data.to_csv("datasets/cardiovascular_train.csv", mode="w", index=False)



preprocessing_one_hot()