from src.entity.config_entity.config_entity import (Data_Ingestion_Config,
                                                    Data_Validation_Config,
                                                    Data_Transformation_Config,
                                                    Model_Train_Config,
                                                    ModelEvaluationConfig
)
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity.config_entity import Training_Pipeline_Config

from src.components.data_validation import Data_Validation

from src.components.data_transformation import Data_transformation

from src.components.model_train import ModelTrainer

from src.components.model_evaluation import ModelEvaluation

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




        data_transformation_config = Data_Transformation_Config(training_pipeline_config=train_pipeline_congig)
        data_transformation = Data_transformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)

        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)


        model_trainer_config = Model_Train_Config(training_pipeline_config=train_pipeline_congig)
        model_trainer = ModelTrainer(model_train_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        print(model_trainer_artifact)



        model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=train_pipeline_congig)
        model_evaluation= ModelEvaluation(model_evaluation_config=model_evaluation_config,data_transformation_artifact=data_transformation_artifact,model_train_artifact=model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.evaluate()
        print(model_evaluation)

    except Exception as e:
        logging.error("Error in main.py")
        raise CustomerException(e,sys)    
        