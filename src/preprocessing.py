"""
Preprocess the data and write the cleaned data to a csv file

Functions
---------
_preprocess_columns(data: pd.DataFrame, columns_to_drop: list, category_columns: list) -> pd.DataFrame
    Drop irrelevant columns, change the datatype of categorical columns and change the labels of columns
_change_labels(data: pd.DataFrame, config_data: dict)
    Change labels of columns to more understandable labels as per the data dictionary
_one_hot_encoding(data: pd.DataFrame) -> pd.DataFrame
    Convert the datatype of categorical columns and Create dummy variables for categorical columns
preprocessor(config_data: dict)
    Acts as the main function for preprocessing the data
"""

import logging

import pandas as pd

from logger import setup_logging
from utils import read_csv, read_yaml, write_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _preprocess_columns(
    data: pd.DataFrame, columns_to_drop: list, category_columns: list
) -> pd.DataFrame:
    """
    Drop irrelevant columns, change the datatype of categorical columns and change the labels of columns

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed

    Returns
    -------
    pandas.DataFrame
        The data with irrelevant columns dropped
    """

    logging.info("Preprocessing columns")

    data = data.drop(columns=columns_to_drop, axis=1)
    data = data.astype({columns: "category" for columns in category_columns})
    return data


def _change_labels(data: pd.DataFrame, config_data: dict):
    """
    Change labels of columns to more understandable labels as per the data dictionary

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed
    """

    logging.info("Changing column labels to more understandable labels")

    column_mappings = {
        "weekday": config_data["weekday_labels"],
        "weathersit": config_data["weathersit_labels"],
        "mnth": config_data["mnth_labels"],
        "season": config_data["season_labels"],
    }

    for col, mapping in column_mappings.items():
        if col in data.columns:
            data[col] = data[col].map(lambda x: mapping.get(x, x))

    return data


def _one_hot_encoding(data: pd.DataFrame, category_columns: list) -> pd.DataFrame:
    """
    Convert the datatype of categorical columns and Create dummy variables for categorical columns

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed

    Returns
    -------
    pandas.DataFrame
        The data with dummy variables created
    """

    logging.info("Creating dummy variables")

    data = pd.get_dummies(data, columns=category_columns, drop_first=True)

    for col in data.select_dtypes(include=["bool"]).columns:
        data[col] = data[col].astype(int)

    return data


def preprocessor(config_data: dict):
    """
    This preprocessor function reads the raw data, drops irrelevant columns, changes labels of columns, creates dummy variables and writes the cleaned data to a csv file
    """

    logging.info("Preprocessing the data")

    data = read_csv(config_data["raw_data_path"])
    logging.info(f"The shape of the data is : {data.shape}")

    preprocessed_data = _preprocess_columns(
        data, config_data["columns_to_drop"], config_data["category_columns"]
    )

    labelled_data = _change_labels(preprocessed_data, config_data)

    encoded_data = _one_hot_encoding(labelled_data, config_data["category_columns"])
    logging.info(f"The shape of the cleaned data is : {encoded_data.shape}")

    write_csv(encoded_data, config_data["cleaned_data_path"])
