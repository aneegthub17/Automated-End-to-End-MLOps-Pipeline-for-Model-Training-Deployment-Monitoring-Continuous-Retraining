from src.data.validator import DataValidator
from src.config import DATASET_PATH
from src.data.loader import DataLoader
from src.eda.profiler import DataProfiler
from src.preprocessing.pipeline import DataPreprocessor

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


def main():

    # ==================================================
    # PIPELINE HEADER
    # ==================================================

    print("=" * 50)
    print(" Enterprise MLOps Pipeline ")
    print("=" * 50)

    # ==================================================
    # 1. DATA INGESTION
    # ==================================================

    print("\n[1/7] DATA INGESTION")

    loader = DataLoader(DATASET_PATH)

    df = loader.load()

    print("✓ Dataset loaded successfully.")

    # ==================================================
    # 2. DATA VALIDATION
    # ==================================================

    print("\n[2/7] DATA VALIDATION")

    validator = DataValidator(df)

    validator.validate()

    # ==================================================
    # 3. DATA PROFILING
    # ==================================================

    print("\n[3/7] DATA PROFILING")

    profiler = DataProfiler(df)

    profiler.profile()

    # ==================================================
    # 4. TRAIN / TEST SPLIT
    # ==================================================

    print("\n[4/7] DATA SPLITTING")

    preprocessor = DataPreprocessor(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = preprocessor.split_data()

    print(
        f"Training Samples : {X_train.shape[0]}"
    )

    print(
        f"Testing Samples  : {X_test.shape[0]}"
    )

    print(
        f"Training Features: {X_train.shape[1]}"
    )

    print(
        f"Testing Features : {X_test.shape[1]}"
    )

    # ==================================================
    # 5. PREPROCESSING
    # ==================================================

    print("\n[5/7] DATA PREPROCESSING")

    (
        X_train_processed,
        X_test_processed,
        preprocessing_pipeline
    ) = preprocessor.preprocess(
        X_train,
        X_test
    )

    print(
        "Processed Training Shape : "
        f"{X_train_processed.shape}"
    )

    print(
        "Processed Testing Shape  : "
        f"{X_test_processed.shape}"
    )

    print(
        "✓ Preprocessing completed."
    )

    # ==================================================
    # 6. MODEL TRAINING
    # ==================================================

    print("\n[6/7] MODEL TRAINING")

    trainer = ModelTrainer()

    models = trainer.train(
        X_train_processed,
        y_train
    )

    # ==================================================
    # 7. MODEL EVALUATION
    # ==================================================

    print("\n[7/7] MODEL EVALUATION")

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(
        models,
        X_test_processed,
        y_test
    )

    # ==================================================
    # DISPLAY MODEL RESULTS
    # ==================================================

    print("\n")
    print("=" * 60)
    print("                 MODEL RESULTS")
    print("=" * 60)

    for model_name, metrics in results.items():

        print(
            f"\n{model_name.upper()}"
        )

        print("-" * 40)

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

    # ==================================================
    # SELECT BEST MODEL
    # ==================================================

    (
        best_model_name,
        best_model
    ) = evaluator.select_best_model(
        models,
        results,
        metric="f1"
    )

    print("\n")
    print("=" * 60)
    print("                  BEST MODEL")
    print("=" * 60)

    print(
        f"\nModel    : {best_model_name}"
    )

    print(
        f"F1 Score : "
        f"{results[best_model_name]['f1']:.4f}"
    )

    print(
        f"Accuracy : "
        f"{results[best_model_name]['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results[best_model_name]['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{results[best_model_name]['recall']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{results[best_model_name]['roc_auc']:.4f}"
    )

    print("\n")
    print("=" * 60)
    print("       ML TRAINING PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()