from pathlib import Path
import joblib


class ModelRegistry:
    """
    Handles saving and loading production-ready ML models.

    The registry stores:
        1. The preprocessing pipeline
        2. The trained model
        3. Model metadata

    This allows the complete inference workflow to be restored later
    without retraining the model.
    """

    def __init__(self, artifact_directory="artifacts/models"):
        self.artifact_directory = Path(artifact_directory)

        # Create the directory if it does not already exist.
        self.artifact_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_model(
        self,
        model,
        preprocessor,
        model_name,
        metrics=None
    ):
        """
        Save a production-ready model artifact.

        Parameters
        ----------
        model :
            Trained machine learning model.

        preprocessor :
            Fitted preprocessing pipeline.

        model_name : str
            Name of the model being saved.

        metrics : dict, optional
            Evaluation metrics for the model.

        Returns
        -------
        Path
            Location of the saved artifact.
        """

        artifact = {
            "model": model,
            "preprocessor": preprocessor,
            "model_name": model_name,
            "metrics": metrics or {}
        }

        model_path = (
            self.artifact_directory
            / f"{model_name}_production.joblib"
        )

        joblib.dump(
            artifact,
            model_path
        )

        print("\n========== MODEL REGISTRY ==========")
        print(f"✓ Production model saved.")
        print(f"Model      : {model_name}")
        print(f"Artifact   : {model_path}")

        if metrics:
            print("\nProduction Model Metrics")

            for metric_name, value in metrics.items():
                print(
                    f"{metric_name:<12}: {value:.4f}"
                )

        return model_path

    def load_model(self, model_name):
        """
        Load a previously saved production model.

        Parameters
        ----------
        model_name : str
            Name of the production model.

        Returns
        -------
        dict
            Saved model artifact containing:
                - model
                - preprocessor
                - model_name
                - metrics
        """

        model_path = (
            self.artifact_directory
            / f"{model_name}_production.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production model not found: {model_path}"
            )

        artifact = joblib.load(model_path)

        print("\n========== MODEL LOADED ==========")
        print(f"✓ Production model loaded.")
        print(f"Model    : {artifact['model_name']}")
        print(f"Artifact : {model_path}")

        return artifact

    def list_models(self):
        """
        List all production model artifacts.

        Returns
        -------
        list
            List of available model artifact paths.
        """

        models = list(
            self.artifact_directory.glob(
                "*_production.joblib"
            )
        )

        print("\n========== REGISTERED MODELS ==========")

        if not models:
            print("No production models registered.")
            return []

        for model in models:
            print(f"✓ {model.name}")

        return models