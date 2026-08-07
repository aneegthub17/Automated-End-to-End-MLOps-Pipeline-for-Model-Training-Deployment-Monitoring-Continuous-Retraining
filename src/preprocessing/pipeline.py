import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, RANDOM_STATE


class DataPreprocessor:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def split_features_target(self):

     X = self.df.drop(
        columns=[
            TARGET_COLUMN,
            "customerID"
            ]
    )

     y = self.df[TARGET_COLUMN]

     return X, y

    def identify_columns(self, X):

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        return numerical_columns, categorical_columns
    def build_pipeline(self, numerical_columns, categorical_columns):
        numerical_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )
        preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numerical_pipeline,
                    numerical_columns
                ),

                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
            
        )
        return preprocessor
    def preprocess(self):
        X, y = self.split_features_target()
        numerical_columns, categorical_columns = self.identify_columns(X)
        preprocessor = self.build_pipeline(
            numerical_columns,
            categorical_columns
        )
        X_processed = preprocessor.fit_transform(X)
        return (
            X_processed,
            y,
            preprocessor
        )


















    
