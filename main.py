import sys
from cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException


STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    error = CustomException(e, sys)
    logging.error(error.error_message)
    raise e