from dataclasses import dataclass

@dataclass
class Data_ingestion_artifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class Data_validation_artifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str    


@dataclass
class Data_transformation_artifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    preprocessed_object_file_path:str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score:float    


@dataclass
class Model_train_artifact:
    trained_model_file_path:str
    train_metric_artifact:float
    test_metric_artifact:float
    best_model_name: str   