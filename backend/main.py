from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np
import os

app = FastAPI (title="Linear Regression API")

#import os
#Model_path = Path(os...)
#model_path = r"G:\IIHMR\Notes\Vinay Sir\MLOps_Pipeline\backend\models\model.pkl"
model_path = Path(os.getenv("model_path", "models/model.pkl"))

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

class InputData(BaseModel):
    area: float
    bedrooms: int

@app.get("/")
def healt_check():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"Predicted_price": float(prediction)}

