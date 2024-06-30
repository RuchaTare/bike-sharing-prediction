"""
Main module for the project
"""

import logging
import os

from logger import setup_logging
from preprocessing import preprocessor
from model_training import trainer
from utils import read_yaml


def main():
    """
    Main function
    """
    setup_logging()
    logging.info("Application started")

    config_data = read_yaml("config.yaml")

    preprocessor(config_data)

    trainer(config_data)


if __name__ == "__main__":
    main()
