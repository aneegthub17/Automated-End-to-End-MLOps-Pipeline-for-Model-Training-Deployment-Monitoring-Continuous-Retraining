from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Customer Churn Prediction API"
    assert data["status"] == "running"
    assert "model" in data


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model" in data


def test_prediction_endpoint():
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card",
        "MonthlyCharges": 27.43,
        "TotalCharges": 137.15
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data

    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
def test_monitoring_endpoint():

    response = client.get("/monitoring")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "monitoring_active"

    assert "model" in data

    assert "metrics" in data

    assert "total_predictions" in data["metrics"]

    assert "average_churn_probability" in data["metrics"]

    assert "predicted_churn" in data["metrics"]

    assert "predicted_no_churn" in data["metrics"]