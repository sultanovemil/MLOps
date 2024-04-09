#!/usr/bin/python3

import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaler(path):
    '''
    Функция для стандартизации данных из файла
    '''  
    data = pd.read_csv(path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(["target"], axis=1))
    cols = ['сhol_lv', 'sys_bl_pres', 'dias_bl_pres', 'pulse', 'glucose_lv']               
    df = pd.DataFrame(scaled_data, columns=cols)
    df_scaled = pd.concat([df, data['target']], axis=1)
    return  df_scaled


def scale_and_save_data():
    '''
    Функция для сохранения стандартизированных данных
    '''
    train_path = "train/"
    test_path = "test/"
    train_data = standard_scaler(train_path + "train_data.csv")
    test_data = standard_scaler(test_path + "test_data.csv")
    train_data.to_csv(train_path + "train_scaled_data.csv", index=False)
    test_data.to_csv(test_path + "test_scaled_data.csv", index=False)


scale_and_save_data()