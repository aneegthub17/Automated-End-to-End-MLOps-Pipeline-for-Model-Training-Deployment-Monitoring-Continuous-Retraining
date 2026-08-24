from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DRIFT DETECTION CONFIGURATION
# ============================================================

NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

PSI_STABLE_THRESHOLD = 0.10
PSI_MODERATE_THRESHOLD = 0.25


# ============================================================
# DATA DRIFT DETECTOR
# ============================================================

class DataDriftDetector:

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ):
        self.reference_data = reference_data.copy()
        self.current_data = current_data.copy()

    # ========================================================
    # VALIDATE INPUT DATA
    # ========================================================

    def validate_data(self):

        missing_reference = [
            column
            for column in NUMERICAL_COLUMNS
            if column not in self.reference_data.columns
        ]

        missing_current = [
            column
            for column in NUMERICAL_COLUMNS
            if column not in self.current_data.columns
        ]

        if missing_reference:
            raise ValueError(
                "Reference dataset is missing columns: "
                f"{missing_reference}"
            )

        if missing_current:
            raise ValueError(
                "Current dataset is missing columns: "
                f"{missing_current}"
            )

        if self.reference_data.empty:
            raise ValueError(
                "Reference dataset contains zero rows."
            )

        if self.current_data.empty:
            raise ValueError(
                "Current dataset contains zero rows."
            )

    # ========================================================
    # CALCULATE PSI
    # ========================================================

    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ):

        reference = pd.to_numeric(
            reference,
            errors="coerce"
        ).dropna()

        current = pd.to_numeric(
            current,
            errors="coerce"
        ).dropna()

        if reference.empty or current.empty:
            return 0.0

        # ----------------------------------------------------
        # Create bins using reference distribution
        # ----------------------------------------------------

        if reference.nunique() < 2:
            return 0.0

        quantile_edges = np.linspace(
            0,
            1,
            bins + 1
        )

        bin_edges = np.quantile(
            reference,
            quantile_edges
        )

        bin_edges = np.unique(
            bin_edges
        )

        if len(bin_edges) < 2:
            return 0.0

        # ----------------------------------------------------
        # Expand first and last boundaries
        # ----------------------------------------------------

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # ----------------------------------------------------
        # Calculate distributions
        # ----------------------------------------------------

        reference_distribution = pd.cut(
            reference,
            bins=bin_edges,
            include_lowest=True
        ).value_counts(
            normalize=True,
            sort=False
        )

        current_distribution = pd.cut(
            current,
            bins=bin_edges,
            include_lowest=True
        ).value_counts(
            normalize=True,
            sort=False
        )

        # ----------------------------------------------------
        # Avoid division by zero
        # ----------------------------------------------------

        epsilon = 0.0001

        reference_distribution = (
            reference_distribution
            .clip(lower=epsilon)
        )

        current_distribution = (
            current_distribution
            .clip(lower=epsilon)
        )

        # ----------------------------------------------------
        # Population Stability Index
        # ----------------------------------------------------

        psi = (
            (
                current_distribution
                - reference_distribution
            )
            *
            np.log(
                current_distribution
                /
                reference_distribution
            )
        ).sum()

        return float(psi)

    # ========================================================
    # CLASSIFY DRIFT
    # ========================================================

    def classify_drift(self, psi):

        if psi < PSI_STABLE_THRESHOLD:

            return "Stable"

        if psi < PSI_MODERATE_THRESHOLD:

            return "Moderate Drift"

        return "Significant Drift"

    # ========================================================
    # ANALYZE NUMERICAL DRIFT
    # ========================================================

    def analyze(self):

        self.validate_data()

        results = []

        for column in NUMERICAL_COLUMNS:

            psi = self.calculate_psi(
                self.reference_data[column],
                self.current_data[column]
            )

            status = self.classify_drift(
                psi
            )

            results.append(
                {
                    "feature": column,
                    "psi": round(psi, 4),
                    "status": status
                }
            )

        results_df = pd.DataFrame(
            results
        )

        significant_drift = int(
            (
                results_df["status"]
                == "Significant Drift"
            ).sum()
        )

        moderate_drift = int(
            (
                results_df["status"]
                == "Moderate Drift"
            ).sum()
        )

        stable_features = int(
            (
                results_df["status"]
                == "Stable"
            ).sum()
        )

        if significant_drift > 0:

            overall_status = "Significant Drift"

        elif moderate_drift > 0:

            overall_status = "Moderate Drift"

        else:

            overall_status = "Stable"

        return {
            "overall_status": overall_status,
            "total_features": len(
                NUMERICAL_COLUMNS
            ),
            "stable_features": stable_features,
            "moderate_drift_features": moderate_drift,
            "significant_drift_features": (
                significant_drift
            ),
            "feature_results": results
        }

    # ========================================================
    # PRINT DRIFT REPORT
    # ========================================================

    def print_report(self):

        report = self.analyze()

        print(
            "\n"
            "==================================================\n"
            "              DATA DRIFT REPORT\n"
            "=================================================="
        )

        print(
            f"Overall Status        : "
            f"{report['overall_status']}"
        )

        print(
            f"Total Features        : "
            f"{report['total_features']}"
        )

        print(
            f"Stable Features       : "
            f"{report['stable_features']}"
        )

        print(
            f"Moderate Drift        : "
            f"{report['moderate_drift_features']}"
        )

        print(
            f"Significant Drift     : "
            f"{report['significant_drift_features']}"
        )

        print(
            "\nFeature Drift Details"
        )

        print(
            "-" * 50
        )

        for result in report["feature_results"]:

            print(
                f"{result['feature']:<20}"
                f" PSI: {result['psi']:<8}"
                f" Status: {result['status']}"
            )

        print(
            "=" * 50
        )


# ============================================================
# TEST DRIFT DETECTION
# ============================================================

if __name__ == "__main__":

    from src.config import DATASET_PATH

    reference_dataset = pd.read_csv(
        DATASET_PATH
    )

    # For initial testing, use the same dataset
    # as both reference and current data.
    current_dataset = reference_dataset.copy()

    detector = DataDriftDetector(
        reference_data=reference_dataset,
        current_data=current_dataset
    )

    detector.print_report()