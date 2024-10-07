"""
This file contains utility functions that are used in the project.
"""

import pandas as pd
import logging
import yaml
import os


def read_yaml(file_path: str) -> dict:
    """
    Read yaml file and return the content

    Parameters
    ----------
    file_path : str
        The path to the yaml file

    Returns
    -------
    dict
        The content of the yaml file
    """

    logging.info("Reading config file")
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read csv file and return a pandas dataframe

    Parameters
    ----------
    file_path : str
        The path to the csv file

    Returns
    -------
    pandas.DataFrame
        The data from the csv file
    """

    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path, engine="python", encoding="utf-8")
    logging.debug(f"shape of the data is {data.shape}")
    return data


def write_csv(data: pd.DataFrame, file_path: str):
    """
    Write the data to a csv file

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be written to the csv file
    file_path : str
        The path to the csv file
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    logging.info(f"Writing data to {file_path}")
    data.to_csv(file_path, index=False)
