import os
from os import path
import sklearn
from joblib import load
from src.load_data import fetch_housing_data,load_housing_data
from src.split_train_test import split
from src.preprocessing import preprocessing_data
from src.train import train
from src.score import score
from src.config import Config
from src.logging_config import configure_logger


def test_model_return_object():
    score = score()
    # =================================
    # TEST SUITE
    # =================================
    # check the return object type
    assert isinstance(score, dict)
    # check the parameter length of the returned object
    assert len(score) == 2
    # check the correctness of parameters returned by the object
    assert "Train-score" in score and "Test-score" in score
    # checking the type of the scores returned by object
    assert isinstance(score["Test-score"], float)
    assert isinstance(score["Train-score"], float)
    # checking the correctness range of returned scores
    assert 0 <= score["Test-score"] <= 1
    assert 0 <= score["Train-score"] <= 1


def test_model_save_load():
    train()
    model_location = Config.model_path
    model_path = os.path.join(model_location, "lin_reg.pickle")
    # =================================
    # TEST SUITE
    # =================================
    # check whether the model file is saved/created in destination directory
    os.path.exists(model_path)
    # check that the model file can be loaded properly
    loaded_model = load(model_path)
    assert isinstance(loaded_model, sklearn.linear_model._base.LinearRegression)


def test_dataset_download():
    fetch_housing_data()
    load_housing_data()
    preprocessing_data()

    train_csv_path = os.path.join(Config.dataset_path, "train.csv")
    val_csv_path = os.path.join(Config.dataset_path, "val.csv")
    train_preprocessed_path = os.path.join(
        Config.processed_dataset_path, "preprocessed_test.csv"
    )
    test_preprocessed_path = os.path.join(
        Config.processed_dataset_path, "preprocessed_train.csv"
    )

    # =================================
    # TEST SUITE
    # =================================
    # Check the datasets file is created/saved in the directory
    assert path.exists(train_csv_path)
    assert path.exists(val_csv_path)
    assert path.exists(train_preprocessed_path)
    assert path.exists(test_preprocessed_path)


test_model_return_object()
test_model_save_load()
test_dataset_download()
