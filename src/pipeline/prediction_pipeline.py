import sys
import pandas as pd
import numpy as np

from src.exception.exception import CustomerException
from src.logging.logging import logging

from src.utils.utils import load_object
from src.constant.constans import TARGET_COLUMN


class PredictionPipeline:

    def __init__(self,model_path:str,preprocessor_path:str):
        try:
            
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)


        except Exception as e:
            raise CustomerException(e,sys)
        

    def _clean_input_data(self,df:pd.DataFrame)->pd.DataFrame:

        """
            Applies the same cleaning logic used during training

        """ 

        try:
            
            logging.info("Cleaning input data for prediction")

            df.columns = df.columns.str.strip()

            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"],axis=1)

            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"], errors="coerce"
                )   

            return df     


        except Exception as e:
            raise CustomerException(e,sys)   
        

    def predict(self,input_df:pd.DataFrame)->pd.DataFrame:
        try:

            logging.info("Starting prediction")

            if TARGET_COLUMN in input_df.columns:
                input_df = input_df.drop(columns=[TARGET_COLUMN])


            cleaned_df = self._clean_input_data(input_df)


            transformed_data = self.preprocessor.transform(cleaned_df)
            predictions = self.model.predict(transformed_data)

            prediction_df = input_df.copy()

            prediction_df["churn_prediction"] = predictions

            prediction_df["churn_prediction_label"] = prediction_df[
                "churn_prediction"
            ].map({1: "Yes", 0: "No"})

            logging.info("Prediction completed successfully")

            return prediction_df
            
        except Exception as e:
            raise CustomerException(e,sys)    
        