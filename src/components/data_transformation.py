
from src.exception.exception import CustomerException
from src.logging.logging import logging
from src.entity.artifacts_entity.artifacat_entity import Data_validation_artifact,Data_transformation_artifact
from src.entity.config_entity.config_entity import Data_Transformation_Config
from src.utils.utils import read_yaml_file
from src.constant.constans import SCHEME_FILE_PATH
from src.utils.utils import save_numpy_array_data,save_object

import sys
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Data_transformation:

    def __init__(self,data_transformation_config:Data_Transformation_Config,
                 data_validation_artifact:Data_validation_artifact):
        
        try:

            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEME_FILE_PATH)

            self.numerical_columns = self._schema_config['numerical_columns']
            self.categorical_columns = self._schema_config['categorical_columns']

        except Exception as e:
            raise CustomerException(e,sys)
        

    def get_preprocessor(self)->ColumnTransformer:

        try:
            
            logging.info("Creating preprocessing pipelines for numerical and categorical data")

            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,self.numerical_columns),
                    ('cat_pipeline',cat_pipeline,self.categorical_columns)
                ]
            )

            return preprocessor



        except Exception as e:
            raise CustomerException(e,sys)



    def clean_dataframe(self,df:pd.DataFrame)->pd.DataFrame:

        try:
            
            logging.info("Cleaning Dataframe")

            # Normalize column names (REAL-WORLD PRACTICE)
            df.columns = df.columns.str.strip()

            # Drop the identifier column
            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"],axis=1)


            # fix totalchanges datatype
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

            return df
                
        except Exception as e:
            raise CustomerException(e,sys)    

    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)


        except Exception as e:
            raise CustomerException(e,sys)       
        


        


    def initiate_data_transformation(self)->Data_transformation_artifact:
        try: 

            self.valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            self.valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = Data_transformation.read_data(self.valid_train_file_path)
            test_df = Data_transformation.read_data(self.valid_test_file_path)

            # split the train features independent and dependent

            self.input_train_features_df = train_df.drop(columns=["Churn"],axis=1)
            self.target_train_column = train_df["Churn"]      

            # split the test features independent and dependent
            self.input_test_features_df = test_df.drop(columns=["Churn"],axis=1)
            self.target_test_column = test_df["Churn"]      
            
            self.target_train_column = (
                self.target_train_column
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"no": 0, "yes": 1})
            )
            self.target_test_column = (
                self.target_test_column
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"no": 0, "yes": 1})
            )


 
            

            if self.target_train_column.isnull().any():
                raise CustomerException("Invalid label values found in training data", sys)

            if self.target_test_column.isnull().any():
                raise CustomerException("Invalid label values found in test data", sys)


            self.cleaned_features_train = self.clean_dataframe(self.input_train_features_df)
            self.cleaned_features_test = self.clean_dataframe(self.input_test_features_df)  

            self.numerical_columns = [col for col in self.numerical_columns if col in self.cleaned_features_train.columns]
            self.categorical_columns = [col for col in self.categorical_columns if col in self.cleaned_features_train.columns]

            self.preprocessor = self.get_preprocessor()

            logging.info(f"columns :{self.cleaned_features_train.columns} ")



            # transformed the data

            self.X_train_transformed = self.preprocessor.fit_transform(self.cleaned_features_train)
            self.X_test_transformed = self.preprocessor.transform(self.cleaned_features_test)


            # concatenate input and target feature array

            train_arr = np.c_[self.X_train_transformed,np.array(self.target_train_column)]
            test_arr = np.c_[self.X_test_transformed,np.array(self.target_test_column)]


            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)


            # save the preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path,self.preprocessor)

            data_transforemed_artifact = Data_transformation_artifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.transformed_object_file_path

            )


            return data_transforemed_artifact

        except Exception as e:
            raise CustomerException(e,sys)    
