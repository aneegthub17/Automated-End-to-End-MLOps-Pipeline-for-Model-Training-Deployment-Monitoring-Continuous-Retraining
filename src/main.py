from src.data.validator import DataValidator
from src.config import DATASET_PATH
from src.data.loader import DataLoader


def main():
    
    print("=" * 50)
    print(" Enterprise MLOps Pipeline ")
    print("=" * 50)

    loader = DataLoader(DATASET_PATH)

    df = loader.load()
    validator = DataValidator(df)
    validator.validate()

    print("\nDataset Loaded Successfully!")
    # print(f"Rows    : {df.shape[0]}")
    # print(f"Columns : {df.shape[1]}")

    print("\nFirst Five Records\n")
    print(df.head())


if __name__ == "__main__":
    main()