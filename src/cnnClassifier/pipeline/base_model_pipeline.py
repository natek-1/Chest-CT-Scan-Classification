import sys

from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.base_model import BaseModel
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException


STAGE_NAME = "Prepare base model"


class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_full_model()




if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error.error_message)
        raise e

    