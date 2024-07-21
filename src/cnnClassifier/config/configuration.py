from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity  import DataIngestionConfig, BaseModelConfig, TrainingConfig

class ConfigManager:

    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_file=config.local_file)
        
        return data_ingestion_config

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=config.root_dir,
            base_model_filename=config.base_model_filename,
            updated_base_model_filename=config.updated_base_model_filename,
            augmentation=self.params.AUGMENTATION,
            image_size=self.params.IMAGE_SIZE,
            batch_size=self.params.BATCH_SIZE,
            include_top=self.params.INCLUDE_TOP,
            epochs=self.params.EPOCHS,
            classes=self.params.CLASSES,
            weights=self.params.WEIGHTS,
            learning_rate=self.params.LEARNING_RATE

        )

        return base_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training

        training_config = TrainingConfig(
            root_dir=config.root_dir,
            training_data_path=config.training_data_path,
            pretrained_model_path=config.pretrained_model_path,
            model_filename=config.model_filename,
            augmentation=self.params.AUGMENTATION,
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            image_size=self.params.IMAGE_SIZE
        )

        return training_config
    