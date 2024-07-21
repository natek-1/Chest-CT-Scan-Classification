
import os
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf




class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.get_base_model()

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.pretrained_model_path)
    
    @staticmethod
    def save_model(model: tf.keras.Model, path: Path):
        model.save(path)
    
    def _train_valid_generator(self):

        data_generator_kwargs = dict(
            rescale=1./255,
            validation_split = 0.2
        )

        valid_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_generator_kwargs)

        dataflow_kwargs = dict(
            target_size = self.config.image_size[:-1],
            batch_size = self.config.batch_size,
            interpolation="bilinear"
        )

        self.valid_generator = valid_data_generator.flow_from_directory(
            self.config.training_data_path,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.augmentation:
            train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                zoom_range=0.1,
                **data_generator_kwargs
            )
        else:
            train_data_generator = valid_data_generator
        
        self.train_generator = train_data_generator.flow_from_directory(
            directory=self.config.training_data_path,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
    
    def train(self):
        train_steps = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        history = self.model.fit(
            self.train_generator,
            epochs = self.config.epochs,
            steps_per_epoch = train_steps,
            validation_data=self.valid_generator,
            validation_steps=validation_steps
        )
        path = os.path.join(self.config.root_dir, self.config.model_filename)

        self.save_model(model=self.model, path=path)

        return history


    


