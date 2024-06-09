"""
Preprocess the data

Functions
---------
read_csv(file_path)
    Read csv file and return a pandas dataframe
drop_columns(data)
    Drop irrelevant columns
change_labels(data, config_data)
    Change labels of columns to more understandable labels as per the data dictionary
create_dummies(data)
    Convert the datatype of categorical columns and Create dummy variables for categorical columns
"""

import logging

import pandas as pd

from logger import setup_logging
from utils import read_csv, read_yaml, write_csv


def drop_columns(data):
    """
    Drop irrelevant columns

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed

    Returns
    -------
    pandas.DataFrame
        The data with irrelevant columns dropped
    """

    logging.info("Dropping irrelevant columns")

    data = data.drop(["instant", "dteday", "casual", "registered"], axis=1)
    return data


def change_labels(data, config_data):
    """
    Change labels of columns to more understandable labels as per the data dictionary

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed
    """

    logging.info("Changing labels of columns")

    column_mappings = {
        "weekday": config_data["weekday_labels"],
        "weathersit": config_data["weathersit_labels"],
        "mnth": config_data["mnth_labels"],
        "season": config_data["season_labels"],
    }
    data = data.apply(
        lambda col: col.map(column_mappings[col.name]) if col.name in column_mappings else col
    )


def create_dummies(data):
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

    data["season"] = data["season"].astype("category")
    data["weekday"] = data["weekday"].astype("category")
    data["mnth"] = data["mnth"].astype("category")
    data["weathersit"] = data["weathersit"].astype("category")

    data = pd.get_dummies(data, drop_first=True)
    return data


def main():
    """
    Main function to preprocess the data
    """

    setup_logging()
    logging.info("Application started")

    config_data = read_yaml("../config.yaml")

    data = read_csv("../data/raw_data.csv")
    logging.info(f"The shape of the data is : {data.shape}")
    logging.info(data.head(20))

    data = drop_columns(data)
    logging.info(f"The remainder columns after dropping {data.columns}")

    change_labels(data, config_data)

    cleaned_data = create_dummies(data)

    write_csv(cleaned_data, "../data/cleaned_data.csv")


if __name__ == "__main__":
    main()
