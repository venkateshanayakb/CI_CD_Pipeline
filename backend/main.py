from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import numpy as np
import os
import logging

# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_NAME = "Linear Regression API"
APP_VERSION = "1.0.0"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))

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

# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class InputData(BaseModel):
    area: float = Field(..., gt=0, description="Area in square feet")
    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms")

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {"message": f"{APP_NAME} is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array([[data.area, data.bedrooms]])
        prediction = model.predict(X)[0]

        logger.info(
            f"Prediction made | area={data.area}, bedrooms={data.bedrooms}, "
            f"predicted_price={prediction}"
        )

        return {
            "predicted_price": float(prediction),
            "model_version": APP_VERSION
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
