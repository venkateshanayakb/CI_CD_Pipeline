from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API running"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data
    assert data["model_loaded"] is True


def test_predict_valid_input():
    payload = {
        "total_discharges": 50,
        "provider_state": "NY",
        "drg_definition": "194 - SIMPLE PNEUMONIA & PLEURISY W CC"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_medicare_payment" in data
    assert isinstance(data["predicted_medicare_payment"], (int, float))
    assert data["predicted_medicare_payment"] > 0
    assert data["currency"] == "USD"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "model_name" in data
    assert "performance" in data
