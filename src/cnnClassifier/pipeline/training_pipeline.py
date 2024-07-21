import sys

from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.training import Training
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training._train_valid_generator()
        history = training.train()
        return history

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error.error_message)
        raise error