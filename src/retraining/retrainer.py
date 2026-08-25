from pathlib import Path
from datetime import datetime

from src.monitoring.monitor import get_drift_summary


# ============================================================
# RETRAINING CONFIGURATION
# ============================================================

PRODUCTION_MODEL_DIRECTORY = Path(
    "artifacts/models"
)

SIGNIFICANT_DRIFT_STATUS = (
    "Significant Drift"
)


# ============================================================
# RETRAINING MANAGER
# ============================================================

class RetrainingManager:
    """
    Controls automated model retraining based on
    detected production data drift.
    """

    def __init__(self):

        self.production_directory = (
            PRODUCTION_MODEL_DIRECTORY
        )

    # ========================================================
    # CHECK WHETHER RETRAINING IS REQUIRED
    # ========================================================

    def check_retraining_required(self):

        drift_summary = (
            get_drift_summary()
        )

        status = drift_summary.get(
            "overall_status",
            drift_summary.get(
                "status"
            )
        )

        if status == "insufficient_data":

            return {
                "retraining_required": False,
                "reason": (
                    "Insufficient production "
                    "data for drift detection."
                ),
                "drift_summary": drift_summary
            }

        if status == SIGNIFICANT_DRIFT_STATUS:

            return {
                "retraining_required": True,
                "reason": (
                    "Significant data drift "
                    "detected."
                ),
                "drift_summary": drift_summary
            }

        return {
            "retraining_required": False,
            "reason": (
                "Data drift is below the "
                "retraining threshold."
            ),
            "drift_summary": drift_summary
        }

    # ========================================================
    # GET PRODUCTION MODEL ARTIFACTS
    # ========================================================

    def get_production_models(self):

        if not self.production_directory.exists():

            return []

        return list(
            self.production_directory.glob(
                "*_production.joblib"
            )
        )

    # ========================================================
    # RUN EXISTING TRAINING PIPELINE
    # ========================================================

    def run_training_pipeline(self):

        print(
            "\n"
            "==================================================\n"
            "           AUTOMATED MODEL RETRAINING\n"
            "=================================================="
        )

        print(
            "\n⚠ Significant data drift detected."
        )

        print(
            "Starting existing training pipeline..."
        )

        try:

            from src.main import main

            result = main()

            return {
                "success": True,
                "result": result
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(error)
            }

    # ========================================================
    # VERIFY PRODUCTION ARTIFACT
    # ========================================================

    def verify_production_model(
        self,
        previous_models
    ):

        current_models = (
            self.get_production_models()
        )

        if not current_models:

            return {
                "verified": False,
                "message": (
                    "No production model "
                    "artifact found after "
                    "retraining."
                )
            }

        previous_paths = {
            str(
                path.resolve()
            )
            for path in previous_models
        }

        new_models = [
            path
            for path in current_models
            if str(
                path.resolve()
            ) not in previous_paths
        ]

        if new_models:

            return {
                "verified": True,
                "message": (
                    "New production model "
                    "artifact created."
                ),
                "new_models": [
                    str(path)
                    for path in new_models
                ]
            }

        latest_model = max(
            current_models,
            key=lambda path: path.stat().st_mtime
        )

        return {
            "verified": True,
            "message": (
                "Production model artifact "
                "updated successfully."
            ),
            "latest_model": str(
                latest_model
            )
        }

    # ========================================================
    # EXECUTE RETRAINING WORKFLOW
    # ========================================================

    def execute(self):

        start_time = datetime.now()

        print(
            "\n"
            "==================================================\n"
            "          RETRAINING DECISION ENGINE\n"
            "=================================================="
        )

        # ----------------------------------------------------
        # Check current drift
        # ----------------------------------------------------

        decision = (
            self.check_retraining_required()
        )

        drift_summary = decision[
            "drift_summary"
        ]

        drift_status = drift_summary.get(
            "overall_status",
            drift_summary.get(
                "status"
            )
        )

        print(
            f"\nDrift Status : {drift_status}"
        )

        print(
            f"Decision     : "
            f"{decision['reason']}"
        )

        # ----------------------------------------------------
        # Stop when retraining isn't required
        # ----------------------------------------------------

        if not decision[
            "retraining_required"
        ]:

            print(
                "\n✅ Retraining not required."
            )

            return {
                "status": "not_required",
                "reason": decision[
                    "reason"
                ],
                "drift_summary": drift_summary
            }

        # ----------------------------------------------------
        # Capture current production models
        # ----------------------------------------------------

        previous_models = (
            self.get_production_models()
        )

        print(
            "\nCurrent production models:"
        )

        if previous_models:

            for model in previous_models:

                print(
                    f"  - {model}"
                )

        else:

            print(
                "  No production models found."
            )

        # ----------------------------------------------------
        # Run training
        # ----------------------------------------------------

        training_result = (
            self.run_training_pipeline()
        )

        if not training_result[
            "success"
        ]:

            print(
                "\n❌ Retraining failed."
            )

            return {
                "status": "failed",
                "reason": (
                    "Training pipeline "
                    "failed."
                ),
                "error": training_result[
                    "error"
                ]
            }

        # ----------------------------------------------------
        # Verify production model
        # ----------------------------------------------------

        verification = (
            self.verify_production_model(
                previous_models
            )
        )

        if not verification[
            "verified"
        ]:

            print(
                "\n❌ Retraining completed, "
                "but production artifact "
                "verification failed."
            )

            return {
                "status": "verification_failed",
                "verification": verification
            }

        # ----------------------------------------------------
        # Calculate duration
        # ----------------------------------------------------

        duration = (
            datetime.now()
            - start_time
        ).total_seconds()

        print(
            "\n✅ RETRAINING COMPLETED"
        )

        print(
            f"Duration : {duration:.2f} seconds"
        )

        print(
            verification[
                "message"
            ]
        )

        return {
            "status": "retrained",
            "reason": decision[
                "reason"
            ],
            "duration_seconds": round(
                duration,
                2
            ),
            "drift_summary": drift_summary,
            "verification": verification
        }


# ============================================================
# MAIN
# ============================================================

def main():

    manager = RetrainingManager()

    result = manager.execute()

    print(
        "\n"
        "==================================================\n"
        "              RETRAINING RESULT\n"
        "=================================================="
    )

    print(
        result
    )

    return result


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()