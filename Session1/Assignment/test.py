import os
import pandas as pd


def check_data_file():
    file = 'data.zip'
    assert not os.path.exist(file), "data file is uploaded"


def check_model_file():
    file = 'model.h5'
    assert not os.path.exist(file), "Model file is uploaded"


def check_model_acc():
    df_metrics = pd.read_csv('metrics.csv')
    acc_validation = df_metrics['val_accuracy']
    assert acc_validation > 0.7, "Overall accuracy is less than 70%"


def check_dog_acc():
    df_metrics = pd.read_csv('metrics.csv')
    acc_dog = df_metrics['dog_acc']
    assert acc_dog > 0.7, "Dog accuracy is less than 70%"


def check_cat_acc():
    df_metrics = pd.read_csv('metrics.csv')
    acc_cat = df_metrics['cat_acc']
    assert acc_cat > 0.7, "Cat accuracy is less than 70%"
