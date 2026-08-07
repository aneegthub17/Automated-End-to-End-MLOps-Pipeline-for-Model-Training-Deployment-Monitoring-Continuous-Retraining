import pandas as pd
from sqlalchemy import label


class DataProfiler:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def profile(self):

        print("\n" + "=" * 50)
        print("           DATA PROFILE REPORT")
        print("=" * 50)

        rows, cols = self.df.shape

        print(f"\nRows                : {rows}")
        print(f"Columns             : {cols}")

        duplicate_rows = self.df.duplicated().sum()

        print(f"Duplicate Rows      : {duplicate_rows}")

        missing_values = self.df.isnull().sum().sum()

        print(f"Missing Values      : {missing_values}")

        categorical = self.df.select_dtypes(include="object").columns

        numerical = self.df.select_dtypes(exclude="object").columns

        print(f"Categorical Columns : {len(categorical)}")

        print(f"Numerical Columns   : {len(numerical)}")

        print("\nTarget Distribution")
        print("-" * 50)

        target_distribution = self.df["Churn"].value_counts()

        for label, count in target_distribution.items():
            percentage = (count / len(self.df)) * 100
            print(f"{label:<5}: {count:>3} ({percentage:.2f}%)")