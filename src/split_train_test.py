import argparse
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from src.config import Config
from src.logging_config import configure_logger

DATASET_PATH = Config.dataset_path
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(Config.processed_dataset_path, exist_ok=True)

if DATASET_PATH is None:
    DATASET_PATH = Config.dataset_path

logger = configure_logger()
logger.info("splitting data to train and test set")

def split(data_path=DATASET_PATH):
    csv_path = os.path.join(DATASET_PATH, "housing.csv")
    housing = pd.read_csv(csv_path)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    strat_train_set.to_csv(DATASET_PATH+'/train.csv', index=False)  # ask- why do we keep index as false?
    strat_test_set.to_csv(DATASET_PATH+'/test.csv', index=False)
    # shutil.rmtree(path="datasets")
logger.info("splitting data complete")
if __name__=='__main__':
    # print("splitting data.....")
    # split(DATASET_PATH)
    # print("splitting data complete.....")
    args = parser.parse_args()
    if args.dataset_path :
        DATASET_PATH = args.dataset_path