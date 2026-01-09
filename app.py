import os
import sys
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.exception.exception import CustomerException
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.entity.config_entity.config_entity import PredictionConfig

# ----------------------
# App setup
# ----------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ----------------------
# Load model ONCE (Industry practice)
# ----------------------
try:
    pred_conf = PredictionConfig()
    predictor = PredictionPipeline(
        model_path=pred_conf.MODEL_PATH,
        preprocessor_path=pred_conf.PREPROCESSOR_PATH,
    )
except Exception as e:
    raise RuntimeError("Failed to load model or preprocessor") from e


# ----------------------
# Routes
# ----------------------
@app.get("/")
async def index():
    """Redirect root to docs"""
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict_route(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload CSV → Predict → Show result table
    """
    try:
        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )

        # Read CSV
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )

        # Run prediction
        prediction_df = predictor.predict(df)

        # Save output (optional)
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", "predictions.csv")
        prediction_df.to_csv(output_path, index=False)

        # Render HTML table
        table_html = prediction_df.to_html(
            classes="table table-striped",
            index=False
        )

        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except CustomerException as ce:
        raise HTTPException(status_code=400, detail=str(ce))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
