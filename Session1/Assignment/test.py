import os
import pandas as pd

base_dir = 'Session1/Assignment/'


def get_key_last_value(file, key):
    df_metrics = pd.read_csv(file, header=0)
    bottom = df_metrics.tail(1)
    return bottom[key].iloc[-1]


def test_check_data_file():
    file = base_dir + 'data.zip'
    assert not os.path.exists(file), "data file is uploaded"


def test_check_model_file():
    file = base_dir + 'model.h5'
    assert not os.path.exists(file), "Model file is uploaded"


def test_check_model_acc():
    file = base_dir + 'metrics.csv'
    acc_validation = get_key_last_value(file, 'val_accuracy')
    assert acc_validation > 0.7, "Overall accuracy is less than 70%"


def test_check_dog_acc():
    file = base_dir + 'metrics.csv'
    acc_dog = get_key_last_value(file, 'dog_acc')
    assert acc_dog > 0.7, "Dog accuracy is less than 70%"


def test_check_cat_acc():
    file = base_dir + 'metrics.csv'
    acc_cat = get_key_last_value(file, 'cat_acc')
    assert acc_cat > 0.7, "Cat accuracy is less than 70%"
