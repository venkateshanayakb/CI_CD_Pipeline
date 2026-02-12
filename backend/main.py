from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import os
import logging

# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_NAME = "Medicare Reimbursement Prediction API"
APP_VERSION = "2.0.0"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "backend/models/model.pkl"))
if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found. Train the model before starting API.")

METRICS_PATH = Path(os.getenv("METRICS_PATH", "backend/models/metrics.json"))


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App Initialization
# --------------------------------------------------

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# --------------------------------------------------
# Model Loading
# --------------------------------------------------

model = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

#-------------------------------------------------------
# METRICS LOADING
#-------------------------------------------------------
metrics = None

try:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            import json
            metrics = json.load(f)
        logger.info("Metrics loaded successfully.")
    else:
        logger.warning("Metrics file not found.")
except Exception as e:
    logger.error(f"Failed to load metrics: {e}")


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class InputData(BaseModel):
    total_discharges: int = Field(..., ge=1, description="Number of total discharges")
    provider_state: str = Field(..., description="State abbreviation (e.g., NY, CA)")
    drg_definition: str = Field(..., description="DRG definition string")

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": model is not None
    }

@app.get("/metrics")
def get_metrics():
    if metrics is None:
        raise HTTPException(status_code=404, detail="Metrics not available")

    return {
        "model_name": metrics.get("selected_model"),
        "performance": metrics.get("metrics"),
        "api_version": APP_VERSION
    }

@app.post("/predict")
def predict(data: InputData):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame (must match training feature names)
        input_df = pd.DataFrame([{
            "Total Discharges": data.total_discharges,
            "Provider State": data.provider_state,
            "DRG Definition": data.drg_definition
        }])

        # Model predicts log(payment)
        log_prediction = model.predict(input_df)[0]

        # Convert back to original dollar scale
        prediction = float(np.exp(log_prediction))

        logger.info(
            f"Prediction made | discharges={data.total_discharges}, "
            f"state={data.provider_state}, "
            f"drg={data.drg_definition}, "
            f"predicted_payment={prediction}"
        )

        return {
            "predicted_medicare_payment": prediction,
            "currency": "USD",
            "model_version": APP_VERSION
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
