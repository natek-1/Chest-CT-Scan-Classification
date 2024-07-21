import os, sys
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import BaseModelConfig


class BaseModel:

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None
    
    @staticmethod
    def save_model(model: tf.keras.Model, path: Path):
        model.save(path)

    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.image_size,
            weights=self.config.weights,
            include_top=self.config.include_top,
        )
        path = os.path.join(self.config.root_dir, self.config.base_model_filename)
        self.save_model(model=self.model, path=path)
    


    def _prepare_full_model(self, freeze_all=True, freeze_till=None):

        if freeze_all:
            for layer in self.model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in self.model.layers[:-freeze_till]:
                layer.trainable = False
        
        model_input = tf.keras.layers.Flatten()(self.model.output)
        output = tf.keras.layers.Dense(16, activation="relu")(model_input)
        output = tf.keras.layers.Dense(self.config.classes,
                                       activation="softmax")(output)
        
        full_model = tf.keras.Model(inputs=self.model.input,
                                    outputs=output)
        
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model

    def update_full_model(self):
        self.full_model = self._prepare_full_model()
        path = os.path.join(self.config.root_dir, self.config.updated_base_model_filename)
        self.save_model(model=self.full_model, path=path)
    

