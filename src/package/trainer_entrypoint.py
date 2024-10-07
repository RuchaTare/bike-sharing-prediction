"""
Entrypoint for model training. Calls setup_logging, data ingestion, trainer functions
"""

import os
from utils import read_json
from logger import setup_logging


def entrypoint(config_data: dict) -> None:
    """
    Entrypoint for model training.Calls setup_logging, read_yaml, preprocessor, and trainer functions

    Parameters
    ----------
    config_data : dict
        Configuration data
    """
    dict.update(config_data["logging"], {"module": "trainer"})
    setup_logging(config_data["logging"])

    logging.info("Create Data Ingestion object")
    data_ingestion = DataIngestion(config_data["data_ingestion"])
    model_trainer = Trainer(config_data)
    try:

        model = model_trainer.train()

    except Exception as e:
        logging.error(e, exc_info=True)
        raise e


if __name__ == "__main__":
    config_data = read_json("trainer_config.json")
    entrypoint(config_data)
