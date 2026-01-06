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