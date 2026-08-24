from datetime import datetime
from pathlib import Path
import json


# ==================================================
# MONITORING CONFIGURATION
# ==================================================

MONITORING_DIRECTORY = Path(
    "experiments/monitoring"
)

PREDICTION_LOG_FILE = (
    MONITORING_DIRECTORY /
    "predictions.jsonl"
)


# ==================================================
# INITIALIZE MONITORING DIRECTORY
# ==================================================

def initialize_monitoring():

    MONITORING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


# ==================================================
# LOG PREDICTION
# ==================================================

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
    - model prediction
    - churn probability
    """

    initialize_monitoring()

    prediction_record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "features": features,
        "prediction": str(prediction),
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


# ==================================================
# LOAD PREDICTION LOG
# ==================================================

def load_predictions():
    """
    Load all prediction records from
    the JSONL monitoring file.
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

                # Ignore malformed monitoring
                # records instead of crashing
                continue

    return predictions


# ==================================================
# GENERATE PREDICTION SUMMARY
# ==================================================

def get_prediction_summary():
    """
    Generate aggregate monitoring metrics.

    Returns:
        total_predictions
        average_churn_probability
        predicted_churn
        predicted_no_churn
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

    # ----------------------------------------------
    # Calculate average churn probability
    # ----------------------------------------------

    probabilities = [
        float(
            record.get(
                "churn_probability",
                0.0
            )
        )
        for record in predictions
    ]

    average_churn_probability = (
        sum(probabilities)
        / total_predictions
    )

    # ----------------------------------------------
    # Count predictions
    # ----------------------------------------------

    predicted_churn = sum(
        1
        for record in predictions
        if str(
            record.get("prediction")
        ).lower() == "yes"
    )

    predicted_no_churn = sum(
        1
        for record in predictions
        if str(
            record.get("prediction")
        ).lower() == "no"
    )

    # ----------------------------------------------
    # Return monitoring summary
    # ----------------------------------------------

    return {
        "total_predictions": total_predictions,
        "average_churn_probability": round(
            average_churn_probability,
            4
        ),
        "predicted_churn": predicted_churn,
        "predicted_no_churn": predicted_no_churn
    }


# ==================================================
# PRINT MONITORING SUMMARY
# ==================================================

def print_prediction_summary():

    summary = get_prediction_summary()

    print(
        "\n"
        "==================================================\n"
        "             PREDICTION MONITORING\n"
        "=================================================="
    )

    print(
        f"Total Predictions        : "
        f"{summary['total_predictions']}"
    )

    print(
        f"Average Churn Probability: "
        f"{summary['average_churn_probability']}"
    )

    print(
        f"Predicted Churn          : "
        f"{summary['predicted_churn']}"
    )

    print(
        f"Predicted No Churn       : "
        f"{summary['predicted_no_churn']}"
    )

    print(
        "=================================================="
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print_prediction_summary()