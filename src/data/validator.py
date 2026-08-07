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

        missing = missing[missing > 0]

        if missing.empty:
            print("✓ No missing values found.")
            return

        print("⚠ Missing values detected:\n")
        print(missing)

        raise ValueError("Dataset contains missing values.")
    def validate_duplicates(self):

        print("\n========== Duplicate Validation ==========")

        duplicate_count = self.df.duplicated().sum()

        if duplicate_count == 0:
            print("✓ No duplicate rows found.")
        else:
            print(f"⚠ Duplicate rows found: {duplicate_count}")

            raise ValueError(
             f"Dataset contains {duplicate_count} duplicate rows."
            )
    def validate(self):
         
           # Runs all validation checks in sequencee 
        self.validate_shape()
        self.validate_columns()
        self.validate_missing_values()
        self.validate_duplicates()
        print("\n✅ Data Validation Completed Successfully.")
