"""
Entrypoint for model training. Calls setup_logging, read_yaml, preprocessor, and trainer functions
"""

import logging

from logger import setup_logging
from model_training import trainer
from preprocessing import preprocessor
from model_training import Trainer
from utils import read_yaml


def entrypoint(config_data: dict) -> None:
    """
    Entrypoint for model training.Calls setup_logging, read_yaml, preprocessor, and trainer functions

    Parameters
    ----------
    config_data : dict
        Configuration data
    """

    setup_logging()

    logging.info("Create Data Ingestion object")
    data_ingestion = DataIngestion(config_data["data_ingestion"])
    model_trainer = Trainer(config_data)
    try:

        model = model_trainer.train()

    except Exception as e:
        logging.error(e, exc_info=True)
        raise e


if __name__ == "__main__":
    config_data = read_yaml("config.yaml")
    entrypoint(config_data)
