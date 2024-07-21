import os, sys
import zipfile


import gdown
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        '''Downloads the file provided in the apprioriate url from the config data ingestion path 
        and saved it to location indicated  as per config folder 
        '''

        try:
            dataset_url = self.config.source_url
            os.makedirs(self.config.root_dir, exist_ok=True)
            zip_download_dir = os.path.join(self.config.root_dir, self.config.local_file)

            file_id = self.config.source_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            logging.info("About to start the download process")

            gdown.download(prefix+file_id,zip_download_dir)

            logging.info(f"downloaded the file at {self.config.source_url} into the file {zip_download_dir}")

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        # make sure file is presente
        unzip_path = self.config.root_dir
        os.makedirs(unzip_path, exist_ok=True)
        path_to_unzip = os.path.join(self.config.root_dir, self.config.local_file)
        with zipfile.ZipFile(path_to_unzip, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            

