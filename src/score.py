import argparse
import pandas as pd
import os
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import Config
from src.logging_config import configure_logger
import pickle
# import train

MODEL_PATH = Config.model_path
PROCESSED_DATA_PATH = Config.processed_dataset_path
model_name = 'forest_reg.pickle'

if PROCESSED_DATA_PATH is None:
    PROCESSED_DATA_PATH = Config.processed_dataset_path

if MODEL_PATH is None:
    MODEL_PATH = Config.model_path


logger = configure_logger()
logger.info("evaluating data")

def score(model_path=MODEL_PATH, processed_dataset_path=PROCESSED_DATA_PATH):
    """inferencing the trained model on test data"""

    print("inferencing")
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    train_path = os.path.join(processed_dataset_path, "processed_train.csv")
    train_set = pd.read_csv(train_path)
    train_set_labels = train_set["median_house_value"]
    train_set.drop("median_house_value", axis=1, inplace=True)

    test_path = os.path.join(processed_dataset_path, "processed_test.csv")
    test_set = pd.read_csv(test_path)
    test_set_labels = test_set["median_house_value"]
    test_set.drop("median_house_value", axis=1, inplace=True)

    model = pickle.load(open(model_path + model_name, "rb"))
    test_score = model.score(test_set, test_set_labels)
    train_score = model.score(train_set, train_set_labels)
    print(test_score)

    return {"Train-score": train_score, "Test-score": test_score}

logger.info("evaluating data complete")

if __name__=='__main__':
    # print("scoring data.....")
    # score(MODEL_PATH,PROCESSED_DATA_PATH)
    # print("scoring data complete.....")
    args = parser.parse_args()
    if args.dataset_path and args.processed_data_path :
        DATASET_PATH = args.dataset_path
        PROCESSED_DATA_PATH = args.processed_data_path
    else:
        DATASET_PATH = Config.dataset_path
        PROCESSED_DATA_PATH = Config.processed_dataset_path
