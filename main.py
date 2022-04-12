
from src.load_data import fetch_housing_data,load_housing_data
from src.split_train_test import split
from src.preprocessing import preprocessing_data
from src.train import train
from src.score import score
from src.config import Config
from src.logging_config import configure_logger


fetch_housing_data()
load_housing_data
print("splitting data")
split()
print("preprocessing data")
preprocessing_data()
print("training data")
train()
print("evaluating data")
score()


