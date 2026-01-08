import os
import sys
import json
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.entity.artifacts_entity.artifacat_entity import (
    Data_transformation_artifact,
    Model_train_artifact
)

from src.entity.config_entity.config_entity import ModelEvaluationConfig
from src.entity.artifacts_entity.model_evaluation_artifact import (
    ModelEvaluationArtifact
)

from src.exception.exception import CustomerException
from src.logging.logging import logging
from src.utils.utils import load_object, load_numpy_array_data


class ModelEvaluation:

    def __init__(self,
                 model_evaluation_config:ModelEvaluationConfig,
                 data_transformation_artifact:Data_transformation_artifact,
                 model_train_artifact:Model_train_artifact
    ):
        try:

            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_train_artifact = model_train_artifact

        except Exception as e:
            raise CustomerException(e,sys)
        

    def evaluate(self)->ModelEvaluationArtifact:
        try:
            
            logging.info("Starting model evaluation")

             # Load model
            model = load_object(
                self.model_train_artifact.trained_model_file_path
            )

            # Load test data

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            # Predictions

            y_pred = model.predict(X_test)

            # Probabilities (for AUC)

            if hasattr(model,"predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None


            # Metrics

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)        

            auc = (
                roc_auc_score(y_test, y_prob)
                if y_prob is not None
                else 0.0
            )

            logging.info(
                f"Evaluation Metrics | F1: {f1:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"AUC: {auc:.4f}"
            )

            # Confusion matrix & report

            report = {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc_score": auc,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

            # Save evaluation report

            os.makedirs(
                os.path.dirname(self.model_evaluation_config.evaluation_report_file_path),
                exist_ok=True
            )

            with open(self.model_evaluation_config.evaluation_report_file_path, "w") as f:
                json.dump(report, f, indent=4)


            # Acceptance criteria
            is_accepted = (
                f1 >= self.model_evaluation_config.minimum_f1_score
                and auc >= self.model_evaluation_config.minimum_auc_score
            )    

            logging.info(
                f"Model acceptance status: {is_accepted}"
            )

            return ModelEvaluationArtifact(
                f1_score=f1,
                roc_auc_score=auc,
                precision=precision,
                recall=recall,
                evaluation_report_file_path=self.model_evaluation_config.evaluation_report_file_path,
                is_model_accepted=is_accepted,
            )

        except Exception as e:
            raise CustomerException(e,sys)    

