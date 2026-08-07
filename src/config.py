from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_NAME = "customer_churn_prediction_dataset.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# MLflow
MLFLOW_EXPERIMENT = "Binary Classification Pipeline"

# Random Seed
RANDOM_STATE = 42

# Target Column
TARGET_COLUMN = "Churn"