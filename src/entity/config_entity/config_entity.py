
import pymongo
from dotenv import load_dotenv
import os
from datetime import datetime
from src.constant import constans

import sys

from src.exception.exception import CustomerException
from src.logging.logging import logging

class Training_Pipeline_Config:

    def __init__(self,timestamp:str=datetime.now()):
        try:
            self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
            self.pipeline_name = constans.PIPELINE_NAME
            self.artifact_name = constans.ARTIFACTS_DIR
            self.artifact_dir =  os.path.join(self.artifact_name,self.timestamp)

        except Exception as e:
            raise CustomerException(e,sys)
        

class Data_Ingestion_Config:

    def __init__(self,training_pipeline_config:Training_Pipeline_Config):
        try:
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constans.DATA_INGESTION_DIR_NAME
            )

            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir,
                constans.DATA_INGESTION_FEATURE_STORE_DIR,
                constans.FILE_NAME
            )

            self.train_file_path = os.path.join(
                self.data_ingestion_dir,
                constans.DATA_INGESTION_INGESTED_DIR,
                constans.TRAIN_FILE_NAME
            )

            self.test_file_path = os.path.join(
                self.data_ingestion_dir,
                constans.DATA_INGESTION_INGESTED_DIR,
                constans.TEST_FILE_NAME
            )

            self.db_name = constans.DATA_INGESTION_DB_NAME
            self.collection_name = constans.DATA_INGESTION_COLLECTION_NAME
            self.train_test_split_ratio = constans.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION


        except Exception as e:
            raise CustomerException(e,sys)