from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    CustomerData,
    PredictionResponse
)

from src.models.registry import ModelRegistry

from src.monitoring.monitor import (
    log_prediction,
    get_monitoring_summary
)


# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production ML API for predicting customer churn "
        "using the trained MLOps pipeline."
    ),
    version="1.0.0"
)


# ==================================================
# LOAD PRODUCTION MODEL
# ==================================================

registry = ModelRegistry()

ARTIFACT_DIRECTORY = Path(
    "artifacts/models"
)


def load_production_model():

    models = registry.list_models()

    if not models:

        raise RuntimeError(
            "No production model found. "
            "Run the training pipeline first."
        )

    # Select the most recently modified artifact.
    latest_model = max(
        models,
        key=lambda path: path.stat().st_mtime
    )

    model_name = latest_model.name.replace(
        "_production.joblib",
        ""
    )

    artifact = registry.load_model(
        model_name
    )

    return artifact


try:

    production_artifact = (
        load_production_model()
    )

    model = production_artifact[
        "model"
    ]

    preprocessor = production_artifact[
        "preprocessor"
    ]

    model_name = production_artifact[
        "model_name"
    ]

except Exception as error:

    model = None
    preprocessor = None
    model_name = None

    print(
        f"⚠ Model loading failed: {error}"
    )


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "service": "Customer Churn Prediction API",
        "status": "running",
        "model": model_name
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():

    if model is None or preprocessor is None:

        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": model_name
    }


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerData):

    if model is None or preprocessor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model is not available."
            )
        )

    try:

        # ------------------------------------------
        # Convert validated request into dictionary
        # ------------------------------------------

        customer_data = customer.model_dump()

        # ------------------------------------------
        # Convert dictionary into DataFrame
        # ------------------------------------------

        dataframe = pd.DataFrame(
            [customer_data]
        )

        # ------------------------------------------
        # Apply SAME preprocessing pipeline
        # ------------------------------------------

        processed_data = (
            preprocessor.transform(
                dataframe
            )
        )

        # ------------------------------------------
        # Generate prediction
        # ------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]

        # ------------------------------------------
        # Generate churn probability
        # ------------------------------------------

        probabilities = model.predict_proba(
            processed_data
        )[0]

        yes_probability = 0.0

        for index, class_name in enumerate(
            model.classes_
        ):

            if class_name == "Yes":

                yes_probability = (
                    probabilities[index]
                )

        # ------------------------------------------
        # Normalize prediction values
        # ------------------------------------------

        prediction = str(
            prediction
        )

        yes_probability = round(
            float(
                yes_probability
            ),
            4
        )

        # ------------------------------------------
        # LOG PREDICTION
        # ------------------------------------------

        log_prediction(
            features=customer_data,
            prediction=prediction,
            churn_probability=yes_probability
        )

        # ------------------------------------------
        # Return API response
        # ------------------------------------------

        return PredictionResponse(
            prediction=prediction,
            churn_probability=yes_probability
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            )
        )


# ==================================================
# MONITORING ENDPOINT
# ==================================================

@app.get("/monitoring")
def monitoring():

    try:

        summary = (
            get_monitoring_summary()
        )

        return {
            "status": "monitoring_active",
            "model": model_name,
            "monitoring": summary
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Monitoring failed: "
                f"{str(error)}"
            )
        )