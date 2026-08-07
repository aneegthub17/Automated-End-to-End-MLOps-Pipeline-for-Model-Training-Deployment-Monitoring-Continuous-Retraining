from src.data.validator import DataValidator
from src.config import DATASET_PATH
from src.data.loader import DataLoader
from src.eda.profiler import DataProfiler
from src.preprocessing.pipeline import DataPreprocessor


def main():

    print("=" * 50)
    print(" Enterprise MLOps Pipeline ")
    print("=" * 50)

    loader = DataLoader(DATASET_PATH)

    df = loader.load()

    validator = DataValidator(df)
    validator.validate()

    profiler = DataProfiler(df)
    profiler.profile()

    preprocessor = DataPreprocessor(df)

    X_processed, y, pipeline = preprocessor.preprocess()

    print("\n========== PREPROCESSING PIPELINE ==========")

    print(f"Processed Feature Shape : {X_processed.shape}")

    print(f"Target Shape            : {y.shape}")

    print("\nDataset Loaded Successfully!")

    print("\nFirst Five Records\n")
    print(df.head())


if __name__ == "__main__":
    main()