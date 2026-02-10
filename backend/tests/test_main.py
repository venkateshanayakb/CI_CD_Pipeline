import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import pickle
import numpy as np

from backend.main import app

Client = TestClient(app)


def test_health_check():
    response = Client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API running"}

def test_predict_valid_input():
    payload = {
        "area" : 1200.5,
        "bedrooms": 3,
    }
    response = Client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "API running"}

def test_predict_valid_input():
    payload = {
        "area": 1200.5,
        "bedrooms": 3
    }
    response = Client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0 #assuming house prices are positive
