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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import read_csv, write_csv


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


def _column_transform(
    data: pd.DataFrame, target_column: str, category_columns: list, numerical_columns: list
) -> pd.DataFrame:
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

    logging.info("One hot encoding categorical columns and minmax scaling numerical columns")

    y = data[target_column]
    data = data.drop(columns=["cnt"])

    transform_object = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), category_columns),
            ("num", MinMaxScaler(), numerical_columns),
        ],
        remainder="passthrough",
    )

    transformed_data = transform_object.fit_transform(data)
    feature_names = transform_object.get_feature_names_out()

    if transformed_data.shape[1] != len(feature_names):
        raise ValueError(
            f"Shape mismatch: transformed_data has {transformed_data.shape[1]} columns,but feature_names has {len(feature_names)} elements"
        )
    transformed_data_dense = transformed_data.toarray()
    transformed_data_df = pd.DataFrame(transformed_data_dense, columns=feature_names)
    transformed_data_df[target_column] = y

    return transformed_data_df


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
    logging.info(f"The shape of the preprocessed data is : {preprocessed_data.shape}")

    labelled_data = _change_labels(preprocessed_data, config_data)
    logging.info(f"The shape of the labelled data is : {labelled_data.shape}")

    transformed_data = _column_transform(
        labelled_data,
        config_data["target_column"],
        config_data["category_columns"],
        config_data["numerical_columns"],
    )
    logging.debug(f"The shape of the transformed data is : {transformed_data.shape}")

    write_csv(transformed_data, config_data["cleaned_data_path"])
