import sys

from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error.error_message)
        raise e

