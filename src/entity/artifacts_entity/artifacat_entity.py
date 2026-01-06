from dataclasses import dataclass

@dataclass
class Data_ingestion_artifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str