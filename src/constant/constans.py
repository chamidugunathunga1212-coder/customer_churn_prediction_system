

"""
This module contains constant values used throughout the application.

"""

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