"""
This module contains the logging configurations for the application.

Functions
---------
setup_logging: Setup logging configurations for the application. Writes logs to a file and console.
"""

import logging
import os
from datetime import datetime


def setup_logging(logger_config: dict):
    """
    Setup logging configurations for the application. Writes logs to a file and console.
    """

    if log_to_file:
        log_file = f{logger_config["module"]}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"}
        logs_dir = os.makedirs(os.path.join(os.getcwd(), logger_config["dir"]), exist_ok=True)
        log_file_path = os.path.join(logs_dir, log_file)

    # logging.basicConfig(
    #     filename=log_file_path,
    #     format=logger_config["format"],
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     level=logging.INFO,
    # )

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter(
    #     "[%(asctime)s] - %(levelname)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    # )
    # console_handler.setFormatter(formatter)
    # logging.getLogger("").addHandler(console_handler)
