import os
import sys
import tarfile
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from six.moves import urllib

# logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
fh = logging.FileHandler("logs/ingest_data.log")
fh.setFormatter(f)
logger.addHandler(fh)
logging.disable()

# basic
"""
logging.basicConfig(
    filename="logs/Logs.txt",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_PATH = "C:\\Users\\venkat.sangam\\Desktop\\Assesments\\Assessment_2\\data\\raw"
HOUSING_PATH = input("Enter folder path to save the data:-")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    print("done")
    return pd.read_csv(csv_path)


def train_validate():
    train_val_fol = input("Enter train and validate path to save:-")
    train_val_path = os.path.join(train_val_fol)
    os.chdir(train_val_path)
    train_set.to_csv("train.csv", index=False)
    test_set.to_csv("test.csv", index=False)
    print("Done train")


# download in our local dir
try:
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
except:
    logger.warning("Failed to download from source")

# fecting the data
try:
    housing = load_housing_data()
except:
    logger.warning("Failed to read the from source")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# train and validate
try:
    train_validate()
except:
    logger.warning("Failed to make train and validate date")
