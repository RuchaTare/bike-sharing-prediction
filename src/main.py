"""
Main module for the project
"""

import logging

from logger import setup_logging
from model_training import trainer
from preprocessing import preprocessor
from utils import read_yaml


def main():
    """
    Main function for the project

    Calls setup_logging, read_yaml, preprocessor, and trainer functions
    """
    setup_logging()
    logging.info("Application started")

    config_data = read_yaml("config.yaml")

    preprocessor(config_data)

    trainer(config_data)


if __name__ == "__main__":
    main()
