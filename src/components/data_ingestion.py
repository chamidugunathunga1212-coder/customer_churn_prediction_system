import numpy as np
import pandas as pd
import os
import sys
import pymongo

from src.exception.exception import CustomerException
from src.logging.logging import logging
from src.entity.config_entity.config_entity import Data_Ingestion_Config,Training_Pipeline_Config

from src.entity.artifacts_entity.artifacat_entity import Data_ingestion_artifact

from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URI")

class DataIngestion:

    def __init__(self,data_ingestion_config:Data_Ingestion_Config):
        
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerException(e,sys)
        

    def export_collection_as_dataframe(self)->pd.DataFrame:

        
        """
            Read the dataset from the mongodb and convert it to dataframe

        """

        try:

            logging.info("Exporting data from Mongodb to feature store")

            self.database_name = self.data_ingestion_config.db_name
            self.collection_name = self.data_ingestion_config.collection_name

            self.mongodb_client = pymongo.MongoClient(MONGODB_URL)

            collection = self.mongodb_client[self.database_name][self.collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info("Dataframe created successfully from the Mongodb collection")

            if "_id" in df.columns.tolist():
                df = df.drop(columns=['_id'],axis=True)

            logging.info(f"Dataframe shape : {df.shape}")    

            return df

        except Exception as e:
            logging.error("Error in exporting collection as dataframe")
            raise CustomerException(e,sys)    
        


    def export_data_into_feature_store(self,df:pd.DataFrame)->pd.DataFrame:
        """
            Export the dataframe into feature store

        """ 
        try:
            logging.info("Exporting data into feature store")

            self.feature_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(self.feature_file_path)
            os.makedirs(dir_path,exist_ok=True)

            df.to_csv(self.feature_file_path,index=False,header=True)
            logging.info("Data exported successfully into feature store")

            return df
            
        except Exception as e:
            logging.error("Error in exporting data into feature store")
            raise CustomerException(e,sys)


    def split_data_as_train_test(self,df:pd.DataFrame)->None:
        """
            Split the data into train and test set and save it to the ingested folder

        """ 
        try:
            self.train_file_path = self.data_ingestion_config.train_file_path
            self.test_file_path = self.data_ingestion_config.test_file_path

            logging.info("Splitting data into train and test set")
            train_Set , test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Creating ingested directory if not available")
            dir_path = os.path.dirname(self.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            if "_id" in train_Set.columns.tolist():
                train_Set = train_Set.drop(columns=['_id'],axis=True)
            if "_id" in test_set.columns.tolist():
                test_set = test_set.drop(columns=['_id'],axis=True)

            train_Set.to_csv(self.train_file_path,index=False,header=True)
            
            test_set.to_csv(self.test_file_path,index=False,header=True)

            logging.info("Train and test file created successfully")

        except Exception as e:
            logging.error("Error in splitting data as train and test")
            raise CustomerException(e,sys)   
        

    def initiate_data_ingestion(self)->Data_ingestion_artifact:

        try:    

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = Data_ingestion_artifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            return data_ingestion_artifact


            


        except Exception as e:
            raise CustomerException(e,sys)