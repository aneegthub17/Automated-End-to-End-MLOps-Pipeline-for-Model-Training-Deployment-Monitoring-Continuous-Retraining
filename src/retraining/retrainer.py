from pathlib import Path
from datetime import datetime

from src.monitoring.monitor import get_drift_summary
from src.models.registry import ModelRegistry


# ============================================================
# RETRAINING CONFIGURATION
# ============================================================

PRODUCTION_MODEL_DIRECTORY = Path(
    "artifacts/models"
)

SIGNIFICANT_DRIFT_STATUS = (
    "Significant Drift"
)

F1_COMPARISON_TOLERANCE = 0.0001


# ============================================================
# RETRAINING MANAGER
# ============================================================

class RetrainingManager:
    """
    Controls automated model retraining.

    Workflow:

        1. Detect significant drift
        2. Load current production model
        3. Train candidate model
        4. Evaluate candidate
        5. Compare candidate against production
        6. Promote only when candidate is meaningfully better
        7. Keep production when candidate is equal or worse
    """

    def __init__(self):

        self.registry = ModelRegistry()

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
    # GET PRODUCTION MODELS
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
    # LOAD CURRENT PRODUCTION MODEL
    # ========================================================

    def load_current_production(self):

        models = (
            self.get_production_models()
        )

        if not models:

            return None

        latest_model = max(
            models,
            key=lambda path: path.stat().st_mtime
        )

        model_name = (
            latest_model.name.replace(
                "_production.joblib",
                ""
            )
        )

        artifact = (
            self.registry.load_model(
                model_name
            )
        )

        return {
            "path": latest_model,
            "model_name": model_name,
            "artifact": artifact
        }

    # ========================================================
    # GET CURRENT PRODUCTION F1
    # ========================================================

    def get_production_f1(
        self,
        production
    ):

        if production is None:

            return None

        artifact = production[
            "artifact"
        ]

        metrics = artifact.get(
            "metrics",
            {}
        )

        f1 = metrics.get(
            "f1"
        )

        if f1 is None:

            return None

        return float(
            f1
        )

    # ========================================================
    # TRAIN CANDIDATE MODEL
    # ========================================================

    def train_candidate(self):

        print(
            "\n"
            "==================================================\n"
            "           CANDIDATE MODEL TRAINING\n"
            "=================================================="
        )

        print(
            "\nTraining candidate model..."
        )

        try:

            from src.main import main

            result = main(
                save_production=False
            )

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
    # COMPARE MODELS
    # ========================================================

    def compare_models(
        self,
        production_f1,
        candidate_f1
    ):
        """
        Compare candidate F1 against production F1.

        A candidate must improve F1 by more than the
        configured tolerance to be promoted.

        Example:

            Production = 0.43137
            Candidate  = 0.43140

        Difference:

            0.000027

        Since this is below 0.0001, the models are treated
        as equal and the candidate is rejected.
        """

        if candidate_f1 is None:

            return {
                "candidate_better": False,
                "reason": (
                    "Candidate F1 score is "
                    "not available."
                )
            }

        candidate_f1 = float(
            candidate_f1
        )

        if production_f1 is None:

            return {
                "candidate_better": True,
                "reason": (
                    "No production F1 score "
                    "is available. Candidate "
                    "will be promoted."
                )
            }

        production_f1 = float(
            production_f1
        )

        difference = (
            candidate_f1
            - production_f1
        )

        if difference > (
            F1_COMPARISON_TOLERANCE
        ):

            return {
                "candidate_better": True,
                "reason": (
                    "Candidate F1 score is "
                    "meaningfully better "
                    "than production."
                ),
                "f1_difference": round(
                    difference,
                    6
                )
            }

        if abs(difference) <= (
            F1_COMPARISON_TOLERANCE
        ):

            return {
                "candidate_better": False,
                "reason": (
                    "Candidate F1 score is "
                    "effectively equal to "
                    "production."
                ),
                "f1_difference": round(
                    difference,
                    6
                )
            }

        return {
            "candidate_better": False,
            "reason": (
                "Candidate F1 score is "
                "lower than production."
            ),
            "f1_difference": round(
                difference,
                6
            )
        }

    # ========================================================
    # PROMOTE CANDIDATE MODEL
    # ========================================================

    def promote_candidate(
        self,
        candidate
    ):

        candidate_result = (
            candidate[
                "result"
            ]
        )

        candidate_model = (
            candidate_result[
                "best_model"
            ]
        )

        candidate_model_name = (
            candidate_result[
                "best_model_name"
            ]
        )

        candidate_metrics = (
            candidate_result[
                "best_metrics"
            ]
        )

        candidate_preprocessor = (
            candidate_result[
                "preprocessing_pipeline"
            ]
        )

        print(
            "\n"
            "==================================================\n"
            "             MODEL PROMOTION\n"
            "=================================================="
        )

        print(
            "\nPromoting candidate model..."
        )

        production_path = (
            self.registry.save_model(
                model=candidate_model,
                preprocessor=(
                    candidate_preprocessor
                ),
                model_name=(
                    candidate_model_name
                ),
                metrics=candidate_metrics
            )
        )

        print(
            "\n✅ Candidate model promoted "
            "to production."
        )

        print(
            f"Production Model : "
            f"{candidate_model_name}"
        )

        print(
            f"Production F1    : "
            f"{candidate_metrics['f1']:.4f}"
        )

        print(
            f"Artifact         : "
            f"{production_path}"
        )

        return {
            "promoted": True,
            "model_name": candidate_model_name,
            "metrics": candidate_metrics,
            "production_path": str(
                production_path
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
        # STEP 1 — CHECK DRIFT
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
            f"\nDrift Status : "
            f"{drift_status}"
        )

        print(
            f"Decision     : "
            f"{decision['reason']}"
        )

        # ----------------------------------------------------
        # STEP 2 — STOP IF RETRAINING NOT REQUIRED
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
        # STEP 3 — LOAD CURRENT PRODUCTION
        # ----------------------------------------------------

        production = (
            self.load_current_production()
        )

        production_f1 = (
            self.get_production_f1(
                production
            )
        )

        print(
            "\nCurrent Production Model"
        )

        if production is None:

            print(
                "Model : None"
            )

            print(
                "F1    : None"
            )

        else:

            print(
                f"Model : "
                f"{production['model_name']}"
            )

            print(
                f"F1    : "
                f"{production_f1:.6f}"
            )

        # ----------------------------------------------------
        # STEP 4 — TRAIN CANDIDATE
        # ----------------------------------------------------

        candidate = (
            self.train_candidate()
        )

        if not candidate[
            "success"
        ]:

            print(
                "\n❌ Candidate training failed."
            )

            return {
                "status": "failed",
                "reason": (
                    "Candidate training "
                    "failed."
                ),
                "error": candidate[
                    "error"
                ]
            }

        # ----------------------------------------------------
        # STEP 5 — GET CANDIDATE METRICS
        # ----------------------------------------------------

        candidate_result = (
            candidate[
                "result"
            ]
        )

        candidate_model_name = (
            candidate_result[
                "best_model_name"
            ]
        )

        candidate_metrics = (
            candidate_result[
                "best_metrics"
            ]
        )

        candidate_f1 = (
            float(
                candidate_metrics[
                    "f1"
                ]
            )
        )

        print(
            "\nCandidate Model"
        )

        print(
            f"Model : "
            f"{candidate_model_name}"
        )

        print(
            f"F1    : "
            f"{candidate_f1:.6f}"
        )

        # ----------------------------------------------------
        # STEP 6 — COMPARE
        # ----------------------------------------------------

        comparison = (
            self.compare_models(
                production_f1,
                candidate_f1
            )
        )

        print(
            "\nModel Comparison"
        )

        if production_f1 is not None:

            print(
                f"Production F1 : "
                f"{production_f1:.6f}"
            )

        else:

            print(
                "Production F1 : None"
            )

        print(
            f"Candidate F1  : "
            f"{candidate_f1:.6f}"
        )

        if (
            "f1_difference"
            in comparison
        ):

            print(
                f"Difference    : "
                f"{comparison['f1_difference']:.6f}"
            )

        print(
            f"Decision      : "
            f"{comparison['reason']}"
        )

        # ----------------------------------------------------
        # STEP 7 — PROMOTE ONLY IF BETTER
        # ----------------------------------------------------

        if comparison[
            "candidate_better"
        ]:

            promotion = (
                self.promote_candidate(
                    candidate
                )
            )

            status = "promoted"

        else:

            print(
                "\n❌ Candidate model rejected."
            )

            print(
                "Existing production model "
                "will remain active."
            )

            promotion = {
                "promoted": False,
                "reason": comparison[
                    "reason"
                ]
            }

            status = "rejected"

        # ----------------------------------------------------
        # STEP 8 — FINAL RESULT
        # ----------------------------------------------------

        duration = (
            datetime.now()
            - start_time
        ).total_seconds()

        print(
            "\n"
            "==================================================\n"
            "              RETRAINING RESULT\n"
            "=================================================="
        )

        print(
            f"Status           : "
            f"{status}"
        )

        print(
            f"Duration         : "
            f"{duration:.2f} seconds"
        )

        print(
            f"Production F1    : "
            f"{production_f1}"
        )

        print(
            f"Candidate F1     : "
            f"{candidate_f1:.6f}"
        )

        print(
            f"Candidate Better : "
            f"{comparison['candidate_better']}"
        )

        return {
            "status": status,
            "reason": decision[
                "reason"
            ],
            "duration_seconds": round(
                duration,
                2
            ),
            "drift_summary": drift_summary,
            "production_f1": production_f1,
            "candidate_model": (
                candidate_model_name
            ),
            "candidate_f1": candidate_f1,
            "candidate_better": (
                comparison[
                    "candidate_better"
                ]
            ),
            "comparison_reason": (
                comparison[
                    "reason"
                ]
            ),
            "promotion": promotion
        }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

def main():

    manager = RetrainingManager()

    return manager.execute()


if __name__ == "__main__":

    main()