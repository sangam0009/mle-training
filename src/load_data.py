import argparse
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib

from src.config import Config
from src.logging_config import configure_logger

DATASET_PATH = None
DATASET_PATH = Config.dataset_path
os.makedirs(DATASET_PATH, exist_ok=True)

if DATASET_PATH is None:
    DATASET_PATH = Config.dataset_path


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(DATASET_PATH)
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# if __name__ == "__main__":
#     logger = configure_logger(log_file=Config.logs_path)
#     parser = argparse.ArgumentParser(description="parsing the dataset path")
#     parser.add_argument("--dataset_path", help="path of the dataset")

#     args = parser.parse_args()
#     if args.dataset_path:
#         DATASET_PATH = args.dataset_path
#     else:
#         DATASET_PATH = Config.dataset_path

# PROCESSED_DATA_PATH = Config.processed_dataset_path
# if DATASET_PATH is None:
#     DATASET_PATH = Config.dataset_path

logger = configure_logger()
logger.info("loading the dataset...")

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """ fetching the data using url"""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    """loading the csv data and loading it a dataframe"""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

logger.info("loading of dataset complete")

if __name__=='__main__':
    # print("loading data.....")
    # fetch_housing_data(HOUSING_URL,HOUSING_PATH)
    # load_housing_data(HOUSING_PATH)
    # print("loading data complete")

    logger = configure_logger(log_file=Config.logs_path)
    parser = argparse.ArgumentParser(description="parsing the dataset path")
    parser.add_argument("--dataset_path", help="path of the dataset")

    args = parser.parse_args()
    if args.dataset_path:
        DATASET_PATH = args.dataset_path
    else:
        DATASET_PATH = Config.dataset_path



