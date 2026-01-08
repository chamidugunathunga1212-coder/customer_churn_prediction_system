from dataclasses import dataclass

@dataclass
class ModelEvaluationArtifact:
    f1_score: float
    roc_auc_score: float
    precision: float
    recall: float
    evaluation_report_file_path: str
    is_model_accepted: bool
