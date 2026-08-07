import pandas as pd

from src.config import TARGET_COLUMN


class DataValidator:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def validate_shape(self):

        print("\n========== Dataset Shape ==========")

        rows, cols = self.df.shape

        if rows == 0:
            raise ValueError("Dataset contains zero rows.")

        print(f"Rows    : {rows}")
        print(f"Columns : {cols}")

    def validate_columns(self):

        print("\n========== Column Validation ==========")

        required_columns = [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            TARGET_COLUMN
        ]

        missing_columns = []

        for column in required_columns:

            if column not in self.df.columns:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(
                f"Missing Columns : {missing_columns}"
            )

        print("✓ All required columns exist.")
    def validate_missing_values(self):

        print("\n========== Missing Value Validation ==========")

        missing = self.df.isnull().sum()

        print(missing)