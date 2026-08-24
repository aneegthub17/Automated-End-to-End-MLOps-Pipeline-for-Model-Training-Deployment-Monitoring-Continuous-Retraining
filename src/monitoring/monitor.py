from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from src.config import DATASET_PATH
from src.drift.detector import DataDriftDetector


# ============================================================
# MONITORING CONFIGURATION
# ============================================================

MONITORING_DIRECTORY = Path(
    "experiments/monitoring"
)

PREDICTION_LOG_FILE = (
    MONITORING_DIRECTORY /
    "predictions.jsonl"
)


# ============================================================
# INITIALIZE MONITORING
# ============================================================

def initialize_monitoring():
    """
    Create the monitoring directory if it does not exist.
    """

    MONITORING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOG PREDICTION
# ============================================================

def log_prediction(
    features,
    prediction,
    churn_probability
):
    """
    Store a prediction event in JSON Lines format.

    Each prediction contains:
        - timestamp
        - customer features
        - prediction
        - churn probability
    """

    initialize_monitoring()

    prediction_record = {
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "features": features,
        "prediction": str(
            prediction
        ),
        "churn_probability": float(
            churn_probability
        )
    }

    with open(
        PREDICTION_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                prediction_record
            ) + "\n"
        )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():
    """
    Load all stored prediction records.
    """

    initialize_monitoring()

    if not PREDICTION_LOG_FILE.exists():

        return []

    predictions = []

    with open(
        PREDICTION_LOG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                predictions.append(
                    record
                )

            except json.JSONDecodeError:

                # Ignore malformed records
                # instead of crashing monitoring.
                continue

    return predictions


# ============================================================
# PREDICTION SUMMARY
# ============================================================

def get_prediction_summary():
    """
    Generate aggregate prediction metrics.
    """

    predictions = load_predictions()

    total_predictions = len(
        predictions
    )

    if total_predictions == 0:

        return {
            "total_predictions": 0,
            "average_churn_probability": 0.0,
            "predicted_churn": 0,
            "predicted_no_churn": 0
        }

    probabilities = [
        float(
            record.get(
                "churn_probability",
                0.0
            )
        )
        for record in predictions
    ]

    average_probability = (
        sum(probabilities)
        / total_predictions
    )

    predicted_churn = sum(
        1
        for record in predictions
        if str(
            record.get(
                "prediction",
                ""
            )
        ).lower() == "yes"
    )

    predicted_no_churn = sum(
        1
        for record in predictions
        if str(
            record.get(
                "prediction",
                ""
            )
        ).lower() == "no"
    )

    return {
        "total_predictions": total_predictions,
        "average_churn_probability": round(
            average_probability,
            4
        ),
        "predicted_churn": predicted_churn,
        "predicted_no_churn": predicted_no_churn
    }


# ============================================================
# DATA DRIFT MONITORING
# ============================================================

def get_drift_summary():
    """
    Compare the training/reference dataset with
    production prediction features.

    The original training dataset acts as the
    reference distribution.

    Features received through the prediction API
    act as current production data.
    """

    reference_data = pd.read_csv(
        DATASET_PATH
    )

    predictions = load_predictions()

    if not predictions:

        return {
            "status": "insufficient_data",
            "message": (
                "No production predictions "
                "available for drift analysis."
            )
        }

    production_records = []

    for record in predictions:

        features = record.get(
            "features",
            {}
        )

        if features:

            production_records.append(
                features
            )

    if not production_records:

        return {
            "status": "insufficient_data",
            "message": (
                "Production prediction records "
                "do not contain feature data."
            )
        }

    current_data = pd.DataFrame(
        production_records
    )

    try:

        detector = DataDriftDetector(
            reference_data=reference_data,
            current_data=current_data
        )

        return detector.analyze()

    except ValueError as error:

        return {
            "status": "insufficient_data",
            "message": str(error)
        }


# ============================================================
# COMPLETE MONITORING SUMMARY
# ============================================================

def get_monitoring_summary():
    """
    Return prediction monitoring and
    data drift monitoring together.
    """

    prediction_summary = (
        get_prediction_summary()
    )

    drift_summary = (
        get_drift_summary()
    )

    return {
        "prediction_monitoring": (
            prediction_summary
        ),
        "drift_monitoring": (
            drift_summary
        )
    }


# ============================================================
# PRINT MONITORING REPORT
# ============================================================

def print_monitoring_report():
    """
    Display a complete monitoring report
    in the terminal.
    """

    summary = get_monitoring_summary()

    print(
        "\n"
        "==================================================\n"
        "             MLOPS MONITORING REPORT\n"
        "=================================================="
    )

    # --------------------------------------------------------
    # Prediction Monitoring
    # --------------------------------------------------------

    print(
        "\nPrediction Monitoring"
    )

    print(
        "-" * 50
    )

    prediction = summary[
        "prediction_monitoring"
    ]

    print(
        f"Total Predictions        : "
        f"{prediction['total_predictions']}"
    )

    print(
        f"Average Churn Probability: "
        f"{prediction['average_churn_probability']}"
    )

    print(
        f"Predicted Churn          : "
        f"{prediction['predicted_churn']}"
    )

    print(
        f"Predicted No Churn       : "
        f"{prediction['predicted_no_churn']}"
    )

    # --------------------------------------------------------
    # Drift Monitoring
    # --------------------------------------------------------

    print(
        "\nData Drift Monitoring"
    )

    print(
        "-" * 50
    )

    drift = summary[
        "drift_monitoring"
    ]

    if drift.get("status") == "insufficient_data":

        print(
            "Status: Insufficient production data"
        )

        print(
            f"Message: "
            f"{drift.get('message', '')}"
        )

    else:

        print(
            f"Overall Status           : "
            f"{drift['overall_status']}"
        )

        print(
            f"Stable Features          : "
            f"{drift['stable_features']}"
        )

        print(
            f"Moderate Drift Features  : "
            f"{drift['moderate_drift_features']}"
        )

        print(
            f"Significant Drift        : "
            f"{drift['significant_drift_features']}"
        )

        print(
            "\nFeature Drift Details"
        )

        print(
            "-" * 50
        )

        for result in drift[
            "feature_results"
        ]:

            print(
                f"{result['feature']:<20}"
                f" PSI: {result['psi']:<8}"
                f" Status: {result['status']}"
            )

    print(
        "=" * 50
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_monitoring_report()