from src.data.validator import DataValidator
from src.config import DATASET_PATH
from src.data.loader import DataLoader
from src.eda.profiler import DataProfiler
from src.preprocessing.pipeline import DataPreprocessor

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry

from src.training.mlflow_tracker import MLflowTracker


def main():

    # ==================================================
    # PIPELINE HEADER
    # ==================================================

    print("=" * 60)
    print("             Enterprise MLOps Pipeline")
    print("=" * 60)

    # ==================================================
    # 1. DATA INGESTION
    # ==================================================

    print("\n[1/9] DATA INGESTION")

    loader = DataLoader(
        DATASET_PATH
    )

    df = loader.load()

    print(
        "✓ Dataset loaded successfully."
    )

    # ==================================================
    # 2. DATA VALIDATION
    # ==================================================

    print("\n[2/9] DATA VALIDATION")

    validator = DataValidator(
        df
    )

    validator.validate()

    # ==================================================
    # 3. DATA PROFILING
    # ==================================================

    print("\n[3/9] DATA PROFILING")

    profiler = DataProfiler(
        df
    )

    profiler.profile()

    # ==================================================
    # 4. TRAIN / TEST SPLIT
    # ==================================================

    print("\n[4/9] DATA SPLITTING")

    preprocessor = DataPreprocessor(
        df
    )

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
    # 5. DATA PREPROCESSING
    # ==================================================

    print("\n[5/9] DATA PREPROCESSING")

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

    print("\n[6/9] MODEL TRAINING")

    trainer = ModelTrainer()

    models = trainer.train(
        X_train_processed,
        y_train
    )

    # ==================================================
    # 7. MODEL EVALUATION
    # ==================================================

    print("\n[7/9] MODEL EVALUATION")

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

    best_metrics = results[
        best_model_name
    ]

    print("\n")
    print("=" * 60)
    print("                  BEST MODEL")
    print("=" * 60)

    print(
        f"\nModel     : {best_model_name}"
    )

    print(
        f"F1 Score  : "
        f"{best_metrics['f1']:.4f}"
    )

    print(
        f"Accuracy  : "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{best_metrics['roc_auc']:.4f}"
    )

    # ==================================================
    # 8. MLFLOW EXPERIMENT TRACKING
    # ==================================================

    print("\n[8/9] MLFLOW EXPERIMENT TRACKING")

    tracker = MLflowTracker()

    print(
        "\nExperiment:"
        " Customer Churn Prediction"
    )

    # --------------------------------------------------
    # Log every trained model
    # --------------------------------------------------

    for model_name, model in models.items():

        print(
            f"\nLogging run: {model_name}"
        )

        tracker.start_run(
            model_name
        )

        try:

            tracker.log_model_run(
                model_name,
                model,
                results[model_name]
            )

            print(
                f"✓ {model_name} logged to MLflow."
            )

        finally:

            tracker.end_run()

    # --------------------------------------------------
    # Log the best model
    # --------------------------------------------------

    print(
        "\nLogging best model..."
    )

    tracker.start_run(
        f"best_{best_model_name}"
    )

    try:

        tracker.log_best_model(
            best_model,
            best_model_name,
            best_metrics
        )

        print(
            "✓ Best model logged to MLflow."
        )

    finally:

        tracker.end_run()

    # ==================================================
    # 9. PRODUCTION MODEL REGISTRY
    # ==================================================

    print("\n[9/9] PRODUCTION MODEL REGISTRY")

    registry = ModelRegistry()

    production_model_path = registry.save_model(
        model=best_model,
        preprocessor=preprocessing_pipeline,
        model_name=best_model_name,
        metrics=best_metrics
    )

    # --------------------------------------------------
    # Verify that the artifact can be loaded
    # --------------------------------------------------

    print(
        "\nVerifying production artifact..."
    )

    loaded_artifact = registry.load_model(
        best_model_name
    )

    print(
        "\n✓ Production artifact verification successful."
    )

    print(
        f"Loaded Model : "
        f"{loaded_artifact['model_name']}"
    )

    print(
        "✓ Preprocessor loaded successfully."
    )

    print(
        "✓ Model loaded successfully."
    )

    # ==================================================
    # COMPLETION
    # ==================================================

    print("\n")
    print("=" * 60)
    print(
        "       COMPLETE MLOPS TRAINING PIPELINE"
    )
    print("=" * 60)

    print(
        "\nProduction Artifact:"
    )

    print(
        f"{production_model_path}"
    )

    print(
        "\nPipeline completed successfully."
    )


if __name__ == "__main__":
    main()