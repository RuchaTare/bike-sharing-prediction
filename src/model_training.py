"""
Model training module
"""

import pandas as pd
import sklearn
import logging
from utils import read_yaml, read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


def read_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Read the cleaned data
    """

    logging.info("Reading the cleaned data")

    try:
        data = read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading the cleaned data: {str(e)}")
    return data


def train_test_split(
    data: pd.DataFrame, test_size: float, random_state: int, target_column: str
) -> pd.DataFrame:
    """
    Split the data into training and test set
    """

    logging.info("Splitting the data into training and test set")

    try:
        X = data.drop(columns=[target_column])
        y = data(target_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return
    except Exception as e:
        logging.error(f"Error splitting the data into training and test set: {str(e)}")
        return None, None, None, None


def model_selection(data):
    """
    Train models with CV and select the best model
    """


def rfe(data):
    """
    Perform Recursive Feature Elimination
    """
    pass


def trainer(config_data):
    """
    Train the model
    """
    logging.info("Training the model")

    data = read_cleaned_data(config_data["cleaned_data_path"])
    logging.info(f"The shape of cleaned data is {data.shape}")
    logging.info(data.head(20))

    X_train, X_test, y_train, y_test = train_test_split(
        data, config_data["test_size"], config_data["random_state"], config_data["target_column"]
    )
    if X_train is None:
        logging.error("Failed to split the data")

    models = [
        LinearRegression(),
        Ridge(),
        HuberRegressor(),
        ElasticNetCV(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        ExtraTreesRegressor(),
        GradientBoostingRegressor(),
    ]
