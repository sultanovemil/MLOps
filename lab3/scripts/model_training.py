from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scripts.download_data import download_data
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
import pandas as pd

def calculate_metric(model):
    _, test_set = download_data() 
    X_test, y_test = test_set.drop(columns=['cardio']), test_set['cardio']  
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label='positive')            
    return f1


def model_training():
    train_set, _ = download_data() 
    X_train, y_train = train_set.drop(columns=['cardio']), train_set['cardio']   
   
    num_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',]
    cat_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    num_pipe = Pipeline([
        ('qt', QuantileTransformer(output_distribution="normal")),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer()),     
    ])
    
    cat_pipe = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessors_all = ColumnTransformer(transformers=[
        ('num_p', num_pipe, num_columns),
        ('cat_p', cat_pipe, cat_columns),    
    ])

    pipe_all = Pipeline([
                ('preprocessors', preprocessors_all),
                ('model', RandomForestClassifier(n_estimators=200,
                                      criterion = "gini",
                                      min_samples_split=15,
                                      max_depth=15,
                                      oob_score=True)
                )
    ])
   
    pipe_all.fit(X_train, y_train)
    
    return pipe_all, calculate_metric(pipe_all)

