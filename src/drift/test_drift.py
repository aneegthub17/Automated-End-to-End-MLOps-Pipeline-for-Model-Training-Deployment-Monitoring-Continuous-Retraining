import pandas as pd

from src.config import DATASET_PATH
from src.drift.detector import DataDriftDetector


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

reference_data = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# CREATE SIMULATED PRODUCTION DATA
# ============================================================

current_data = reference_data.copy()


# ============================================================
# SIMULATE PRODUCTION DATA DRIFT
# ============================================================

# Increase MonthlyCharges significantly.
#
# This simulates a real-world situation where
# customers in the production environment are
# paying considerably higher monthly charges than
# the customers used during model training.

current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.8
)


# Increase TotalCharges as well.

current_data["TotalCharges"] = (
    current_data["TotalCharges"] * 1.8
)


# ============================================================
# RUN DRIFT DETECTION
# ============================================================

detector = DataDriftDetector(
    reference_data=reference_data,
    current_data=current_data
)


# ============================================================
# PRINT DRIFT REPORT
# ============================================================

print(
    "\n"
    "==================================================\n"
    "        SIMULATED PRODUCTION DRIFT TEST\n"
    "=================================================="
)

print(
    "\nReference Dataset"
)

print(
    f"Rows: {reference_data.shape[0]}"
)

print(
    f"Columns: {reference_data.shape[1]}"
)

print(
    "\nSimulated Production Dataset"
)

print(
    f"Rows: {current_data.shape[0]}"
)

print(
    f"Columns: {current_data.shape[1]}"
)


detector.print_report()


# ============================================================
# VERIFY DRIFT WAS DETECTED
# ============================================================

report = detector.analyze()


if report["overall_status"] == "Stable":

    print(
        "\n❌ DRIFT TEST FAILED"
    )

    print(
        "The simulated production data did not "
        "produce detectable drift."
    )

else:

    print(
        "\n✅ DRIFT TEST PASSED"
    )

    print(
        "The drift detection system successfully "
        "detected distribution changes."
    )