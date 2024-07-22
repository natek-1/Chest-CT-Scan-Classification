import sys
from pathlib import Path

import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from urllib.parse import urlparse

from cnnClassifier.exception import CustomException
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier.logger import logging
from dotenv import load_dotenv





load_dotenv()

class Evaluation:

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = self.load_model(self.config.model_path)
        self.valid_generator = None
        self._valid_generator()
        self.scores = self._evaluate_model()

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.image_size[:-1],
            batch_size=self.config.batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data_path,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def _evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.valid_generator)
        y_pred_probs = self.model.predict(self.valid_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.valid_generator.classes

        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        scores = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return scores


    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores)

    def login_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model"
                , registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
