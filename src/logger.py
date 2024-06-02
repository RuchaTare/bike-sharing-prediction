"""
Logging module for the application.
"""

import logging
import os
from datetime import datetime


def setup_logging():
    """
    Setup logging configurations for the application. Writes logs to a file and console.
    """

    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[%(asctime)s] - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logging.getLogger("").addHandler(console_handler)
