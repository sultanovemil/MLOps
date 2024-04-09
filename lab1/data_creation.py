#!/usr/bin/python3

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def make_regression_custom(n_samples, noise):
    '''
    # Функция для генерирования медицинских анализов:
    # уровень холестерина в крови, артериальное давление, пульс, уровень глюкозы
    '''
    n_features = 5
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_features,
                           noise=noise,
                           random_state=42)
    feature_ranges = [(200, 240), (120, 160), (80, 100), (60, 100), (70, 200)]
    for i in range(n_features):
        min_val, max_val = feature_ranges[i]
        #X[:, i] = np.abs(X[:, i] * (max_val - min_val) + min_val)
        X[:, i] = X[:, i] * np.random.randint(min_val, max_val)
  
    cols = ['сhol_lv', 'sys_bl_pres', 'dias_bl_pres', 'pulse', 'glucose_lv']
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y   
    return df


def create_and_save_data(num_samples):
    '''
    Функция для создания и сохраниения набора данных для обучения и тестирования
    '''    
    train_url = "train/train_data.csv"
    test_url = "test/test_data.csv"
    data = make_regression_custom(num_samples, 5)
    X, y = data.drop("target", axis=1), data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(train_url, index=False)
    test_data.to_csv(test_url, index=False)


create_and_save_data(num_samples=200)
                     
  