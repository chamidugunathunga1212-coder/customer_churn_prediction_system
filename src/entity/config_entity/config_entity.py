
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
        

class Data_Validation_Config:

    def __init__(self,training_pipeline_config:Training_Pipeline_Config):
        try:
            
            self.data_validation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constans.DATA_VALIDATION_DIR_NAME
            )

            self.valid_data_dir = os.path.join(
                self.data_validation_dir,
                constans.DATA_VALIDATION_VALID_DIR
            )

            self.invalid_data_dir = os.path.join(
                self.data_validation_dir,
                constans.DATA_VALIDATION_INVALID_DIR
            )

            self.valid_train_file_path = os.path.join(
                self.valid_data_dir,
                constans.TRAIN_FILE_NAME
            )

            self.valid_test_file_path = os.path.join(
                self.valid_data_dir,
                constans.TEST_FILE_NAME
            )

            self.invalid_train_file_path = os.path.join(
                self.invalid_data_dir,
                constans.TRAIN_FILE_NAME
            )

            self.invalid_test_file_path = os.path.join(
                self.invalid_data_dir,
                constans.TEST_FILE_NAME
            )

            self.drift_report_file_path = os.path.join(
                self.data_validation_dir,
                constans.DATA_VALIDATION_DRIFT_REPORT_DIR,
                constans.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
            )


        except Exception as e:
            raise CustomerException(e,sys)        
        


class Data_Transformation_Config:

    def __init__(self,training_pipeline_config:Training_Pipeline_Config):
        try:
            
            self.data_transformation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constans.DATA_TRANSFORMATION_DIR_NAME
            )

            self.transformed_data_dir = os.path.join(
                self.data_transformation_dir,
                constans.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            )

            self.transformed_object_dir = os.path.join(
                self.data_transformation_dir,
                constans.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
            )

            self.transformed_train_file_path = os.path.join(
                self.transformed_data_dir,
                constans.DATA_TRANSFORMATION_TRAIN_FILE_PATH,
            )

            self.transformed_test_file_path = os.path.join(
                self.transformed_data_dir,
                constans.DATA_TRANSFORMATION_TEST_FILE_PATH
            )

            self.transformed_object_file_path = os.path.join(
                self.transformed_object_dir,
                constans.PREPROCESSING_OBJECT_FILE_NAME
            )

        except Exception as e:
            raise CustomerException(e,sys)    



class Model_Train_Config:

    def __init__(self,training_pipeline_config:Training_Pipeline_Config):

        try:
            
            self.model_train_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constans.MODEL_TRAINER_DIR_NAME
            )

            self.model_train_file_path = os.path.join(
                self.model_train_dir,
                constans.MODEL_TRAINER_TRAINED_MODEL_DIR,
                constans.MODEL_TRAINER_TRAINED_MODEL_NAME
            )

            self.threshold_accuracy = constans.MODEL_TRAINER_ACCURACY_THRESHOLD


        except Exception as e:
            raise CustomerException(e,sys)
        
class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: Training_Pipeline_Config):
        try:
            self.model_evaluation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constans.MODEL_EVALUATION_DIR_NAME
            )

            self.evaluation_report_file_path = os.path.join(
                self.model_evaluation_dir,
                constans.MODEL_EVALUATION_REPORT_NAME
            )

            self.minimum_f1_score = constans.MODEL_EVALUATION_MIN_F1_SCORE
            self.minimum_auc_score = constans.MODEL_EVALUATION_MIN_AUC_SCORE

        except Exception as e:
            raise CustomerException(e,sys)    
        
class PredictionConfig:
    MODEL_PATH = os.path.join(
        "final_pikels",
        "model.pkl"
    )

    PREPROCESSOR_PATH = os.path.join(
        "final_pikels",
        "preprocessor.pkl"
    )        
