from src.exception.exception import CustomerException
from src.logging.logging import logging


from src.entity.config_entity.config_entity import Data_Validation_Config
from src.entity.artifacts_entity.artifacat_entity import Data_ingestion_artifact,Data_validation_artifact
from src.utils.utils import read_yaml_file

from src.constant.constans import SCHEME_FILE_PATH

from scipy.stats import ks_2samp
from src.utils.utils import write_yaml_file

import sys
import os
import pandas as pd
import numpy as np


class Data_Validation:

    def __init__(self,data_validation_config:Data_Validation_Config,
                 data_ingestion_artifact:Data_ingestion_artifact):
        try:

            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEME_FILE_PATH)


        except Exception as e:
            raise CustomerException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomerException(e,sys)   


    # def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
    #     try:

    #         number_of_columns = len(self._schema_config)
    #         logging.info(f"Required number of columns: {number_of_columns} ")
    #         logging.info(f"Dataframe has columns: {len(dataframe.columns)}")

    #         if len(dataframe.columns) == number_of_columns:
    #             return True
    #         return False
            
    #     except Exception as e:
    #         raise CustomerException(e,sys)      
        

    def validate_column_names(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns = set(self._schema_config["columns"].keys())
            actual_columns = set(df.columns)
            missing = expected_columns - actual_columns
            extra = actual_columns - expected_columns

            if missing:
                logging.error(f"Missing columns: {missing}")
            if extra:
                logging.error(f"Extra columns: {extra}")

            return len(missing) == 0
        except Exception as e:
            raise CustomerException(e, sys)

    def validate_numerical_columns(self, df: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]

            for col in numerical_columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                invalid_count = converted.isna().sum()

                if invalid_count > 0:
                    logging.warning(
                        f"{col} has {invalid_count} invalid numeric values"
                    )

            return True
        except Exception as e:
            raise CustomerException(e, sys)

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)-> bool:
        try:
            
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1,d2)

                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({
                    column: {"P_value": float(is_same_dist.pvalue), "drift_status":is_found}
                })        

                dirft_report_file_path = self.data_validation_config.drift_report_file_path

                ## create directory
                dir_path = os.path.dirname(dirft_report_file_path)
                os.makedirs(dir_path,exist_ok=True)
                write_yaml_file(file_path=dirft_report_file_path,content=report)  
                
            return status

        except Exception as e:
            raise CustomerException(e,sys)                  
        


    def initiate_data_validation(self)->Data_validation_artifact:
        try:
            # retain_file_path and test_file_path
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read data
            train_df = Data_Validation.read_data(train_file_path)
            test_df = Data_Validation.read_data(test_file_path)

            # # validate number of columns
            # status = self.validate_number_of_columns(train_df)
            # logging.info(f"Number of columns train validation completed. Status: {status}")
            # if not status:
            #     raise Exception("Number of columns are not matching in training file")
            # status = self.validate_number_of_columns(test_df)
            # logging.info(f"Number of columns test validation completed. Status: {status}")
            # if not status:
            #     raise Exception("Number of columns are not matching in testing file")
            
            # validate column names
            status = self.validate_column_names(train_df)
            logging.info(f"Column names train validation completed. Status: {status}")
            if not status:
                raise Exception("Column names are not matching in training file")
            status = self.validate_column_names(test_df)
            logging.info(f"Column names test validation completed. Status: {status}")
            if not status:
                raise Exception("Column names are not matching in testing file")

            # validate numerical columns
            status = self.validate_numerical_columns(train_df)
            logging.info(f"Numerical columns train validation completed. Status: {status}")
            if not status:
                raise Exception("Numerical columns are not matching in training file")
            status = self.validate_numerical_columns(test_df)
            logging.info(f"Numerical columns test validation completed. Status: {status}")
            if not status:
                raise Exception("Numerical columns are not matching in testing file")
            
            # detect data drift
            status=self.detect_dataset_drift(base_df=train_df,current_df=test_df)
            logging.info(f"Data drift detection completed. Drift status: {status}")

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)


            data_validation_artifact = Data_validation_artifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact


        except Exception as e:
            raise CustomerException(e,sys)    