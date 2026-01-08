import os

"""
This module contains constant values used throughout the application.

"""

SCHEME_FILE_PATH = os.path.join("data_schema","schema.yaml")

PIPELINE_NAME = "customer_churn_prediction_pipeline"
ARTIFACTS_DIR = "artifacts"
FILE_NAME = "customer_churn_data.csv"

TARGET_COLUMN = "Churn"

NUMERICAL_COLUMNS = [
    "tenure",
    "TotalCharges",
    "MonthlyCharges",
    "SeniorCitizen"
]

"""
     data_ingestion modules needs these constants.
"""

DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.2

DATA_INGESTION_COLLECTION_NAME = "CustomerChurnCollection"
DATA_INGESTION_DB_NAME = "TelcoCustomerChurnDB"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


"""     

data_validation modules needs these constants.

"""

DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR = "validated"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"


"""
data_transformation modules needs these constants.

"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessing_object.pkl"


DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"




"""
    model_training modules needs these constants

"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_ACCURACY_THRESHOLD: float = 0.6

"""
    model_evaluation modules needs these constants

"""

# Model Evaluation
MODEL_EVALUATION_DIR_NAME = "model_evaluation"
MODEL_EVALUATION_REPORT_NAME = "evaluation_report.json"

MODEL_EVALUATION_MIN_F1_SCORE = 0.60
MODEL_EVALUATION_MIN_AUC_SCORE = 0.65