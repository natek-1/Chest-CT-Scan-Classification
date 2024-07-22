import sys

from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException


STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.login_mlflow()
        evaluation.save_score()

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error.error_message)
        raise error