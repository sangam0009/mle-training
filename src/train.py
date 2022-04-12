import argparse
import pandas as pd
import os
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import logging
from src.config import Config
from src.logging_config import configure_logger
import pickle

MODEL_PATH = Config.model_path
PROCESSED_DATA_PATH = Config.processed_dataset_path
os.makedirs(MODEL_PATH, exist_ok=True)

if PROCESSED_DATA_PATH is None:
    PROCESSED_DATA_PATH = Config.processed_dataset_path

if MODEL_PATH is None:
    MODEL_PATH = Config.model_path


logger = configure_logger()
logger.info("training data")

def train(processed_data_path=PROCESSED_DATA_PATH, model_path= MODEL_PATH):
    """ Training different ML models on train data """
    train_preprocessed_path = os.path.join(processed_data_path, "processed_train.csv")
    train_set = pd.read_csv(train_preprocessed_path)

    housing_prepared = train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = train_set["median_house_value"].copy()

    # logger.debug('Training models')

    #linear regression model

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    pickle.dump(lin_reg, open(model_path+'lin_reg.pickle', "wb"))

    #decision tree model

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    pickle.dump(tree_reg, open(model_path+'tree_reg.pickle', "wb"))

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    #random forest model

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    pickle.dump(final_model, open(model_path+'forest_reg.pickle', "wb"))

logger.info("training data complete")
if __name__=='__main__':
    # print("training data.....")
    # train(PROCESSED_DATA_PATH,MODEL_PATH)
    # print("training data complete.....")
    args = parser.parse_args()
    if args.dataset_path and args.processed_data_path :
        DATASET_PATH = args.dataset_path
        PROCESSED_DATA_PATH = args.processed_data_path
    else:
        DATASET_PATH = Config.dataset_path
        PROCESSED_DATA_PATH = Config.processed_dataset_path
