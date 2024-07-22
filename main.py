import sys
from cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from cnnClassifier.pipeline.base_model_pipeline import BaseModelPipeline
from cnnClassifier.pipeline.training_pipeline import ModelTrainingPipeline
from cnnClassifier.pipeline.evaluation_pipeline import EvaluationPipeline
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException


STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    STAGE_NAME = "Prepare base model"
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    base_model = BaseModelPipeline()
    base_model.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    STAGE_NAME = "Training"
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    STAGE_NAME = "Evaluation stage"
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    error = CustomException(e, sys)
    logging.error(error.error_message)
    raise error