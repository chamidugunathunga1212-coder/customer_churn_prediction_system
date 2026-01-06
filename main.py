from src.entity.config_entity.config_entity import Data_Ingestion_Config,Data_Validation_Config
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity.config_entity import Training_Pipeline_Config

from src.components.data_validation import Data_Validation

from src.logging.logging import logging
from src.exception.exception import CustomerException

import sys

if __name__=='__main__':
    try:

        train_pipeline_congig = Training_Pipeline_Config()
        data_ingestion_config = Data_Ingestion_Config(training_pipeline_config=train_pipeline_congig)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)


        data_validation_config = Data_Validation_Config(training_pipeline_config=train_pipeline_congig)
        data_validation = Data_Validation(data_validation_config=data_validation_config,
                                        data_ingestion_artifact=data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)



    except Exception as e:
        logging.error("Error in main.py")
        raise CustomerException(e,sys)    
        