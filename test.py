import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

from src.entity.artifacts_entity.artifacat_entity import Model_train_artifact
from src.entity.artifacts_entity.artifacat_entity import Data_transformation_artifact

input_data = pd.read_csv("data/sample_customer_churn_full_schema.csv")

predictor = PredictionPipeline(
    model_path="final_pikels/model.pkl",
    preprocessor_path="final_pikels/preprocessor.pkl"
)

predictions = predictor.predict(input_data)
predictions.to_csv("data\output_sample.csv",index=False,header=True)
