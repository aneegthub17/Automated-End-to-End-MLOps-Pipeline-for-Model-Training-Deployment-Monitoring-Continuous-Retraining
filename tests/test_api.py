from fastapi.testclient import TestClient

from src.api.main import app


# ==================================================
# TEST CLIENT
# ==================================================

client = TestClient(
    app
)


# ==================================================
# ROOT ENDPOINT TEST
# ==================================================

def test_root_endpoint():

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == (
        "Customer Churn Prediction API"
    )

    assert data["status"] == "running"

    assert "model" in data


# ==================================================
# HEALTH ENDPOINT TEST
# ==================================================

def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert "status" in data

    assert "model_loaded" in data


# ==================================================
# PREDICTION ENDPOINT TEST
# ==================================================

def test_prediction_endpoint():

    customer = {
        "customerID": "TEST001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "churn_probability" in data

    assert data["prediction"] in [
        "Yes",
        "No"
    ]

    assert (
        0
        <= data["churn_probability"]
        <= 1
    )


# ==================================================
# MONITORING ENDPOINT TEST
# ==================================================

def test_monitoring_endpoint():

    response = client.get(
        "/monitoring"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == (
        "monitoring_active"
    )

    assert "model" in data

    assert "monitoring" in data

    monitoring = data[
        "monitoring"
    ]

    assert "prediction_monitoring" in (
        monitoring
    )

    assert "drift_monitoring" in (
        monitoring
    )

    prediction_metrics = (
        monitoring[
            "prediction_monitoring"
        ]
    )

    assert "total_predictions" in (
        prediction_metrics
    )

    assert "average_churn_probability" in (
        prediction_metrics
    )

    assert "predicted_churn" in (
        prediction_metrics
    )

    assert "predicted_no_churn" in (
        prediction_metrics
    )