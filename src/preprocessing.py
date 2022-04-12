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
PROCESSED_DATA_PATH = Config.processed_dataset_path
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

if DATASET_PATH is None:
    DATASET_PATH = Config.dataset_path

if PROCESSED_DATA_PATH is None:
    PROCESSED_DATA_PATH = Config.processed_dataset_path

logger = configure_logger()
logger.info("preprocessing data")

def preprocessing_data(data_path=DATASET_PATH, processed_data_path=PROCESSED_DATA_PATH):
    # logger.info("inside the preprocessing_data function")

    train_csv_path = os.path.join(data_path, "train.csv")
    val_csv_path = os.path.join(data_path, "test.csv")
    train_preprocessed_path = os.path.join(processed_data_path, "preprocessed_housing_train.csv")
    test_preprocessed_path = os.path.join(processed_data_path, "preprocessed_housing_test.csv")
    train_set = pd.read_csv(train_csv_path)
    test_set = pd.read_csv(val_csv_path)
    strat_train_set_num = train_set.drop("ocean_proximity", axis=1)
    strat_test_set_num = test_set.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(strat_train_set_num)
    X_train = imputer.transform(strat_train_set_num)
    X_test = imputer.transform(strat_test_set_num)

    housing_train = pd.DataFrame(
        X_train, columns=strat_train_set_num.columns, index=train_set.index
    )

    housing_test = pd.DataFrame(X_test, columns=strat_test_set_num.columns, index=test_set.index)

    housing_train["rooms_per_house_hold"] = (
        housing_train["total_rooms"] / housing_train["households"]
    )
    housing_test["rooms_per_house_hold"] = housing_test["total_rooms"] / housing_test["households"]

    housing_train["bedrooms_per_room"] = (
        housing_train["total_bedrooms"] / housing_train["total_rooms"]
    )
    housing_test["bedrooms_per_room"] = (
        housing_test["total_bedrooms"] / housing_test["total_rooms"]
    )

    housing_train["population_per_household"] = (
        housing_train["population"] / housing_train["households"]
    )
    housing_test["population_per_household"] = (
        housing_test["population"] / housing_test["households"]
    )

    housing_train_cat = train_set[["ocean_proximity"]]
    housing_test_cat = test_set[["ocean_proximity"]]

    housing_train_prepared = housing_train.join(pd.get_dummies(housing_train_cat, drop_first=True))
    housing_test_prepared = housing_test.join(pd.get_dummies(housing_test_cat, drop_first=True))

    housing_train_prepared.to_csv(processed_data_path+'processed_train.csv', index=False)
    housing_test_prepared.to_csv(processed_data_path+'processed_test.csv', index=False)

logger.info("preprocessing data complete")
if __name__=='__main__':
    # print("preprocessing data.....")
    # preprocessing_data(DATASET_PATH,PROCESSED_DATA_PATH)
    # print("preprocessing data complete.....")

    args = parser.parse_args()
    if args.dataset_path and args.processed_data_path :
        DATASET_PATH = args.dataset_path
        PROCESSED_DATA_PATH = args.processed_data_path
    else:
        DATASET_PATH = Config.dataset_path
        PROCESSED_DATA_PATH = Config.processed_dataset_path
