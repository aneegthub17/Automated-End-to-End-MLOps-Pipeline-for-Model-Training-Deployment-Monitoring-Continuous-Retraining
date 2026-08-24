import os

import mlflow
import mlflow.sklearn


class MLflowTracker:

    def __init__(
        self,
        experiment_name="Customer Churn Prediction",
        tracking_uri="sqlite:///mlflow.db"
    ):

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # --------------------------------------------------
        # Configure MLflow
        # --------------------------------------------------

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        mlflow.set_experiment(
            self.experiment_name
        )

    def start_run(
        self,
        model_name
    ):

        run = mlflow.start_run(
            run_name=model_name
        )

        return run

    def log_model_run(
        self,
        model_name,
        model,
        metrics
    ):

        # --------------------------------------------------
        # Log model information
        # --------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        # --------------------------------------------------
        # Log evaluation metrics
        # --------------------------------------------------

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(metric_value)
            )

        # --------------------------------------------------
        # Log model artifact
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

    def end_run(self):

        mlflow.end_run()

    def log_best_model(
        self,
        model,
        model_name,
        metrics
    ):

        # --------------------------------------------------
        # Log best model information
        # --------------------------------------------------

        mlflow.log_param(
            "best_model",
            model_name
        )

        # --------------------------------------------------
        # Log best model metrics
        # --------------------------------------------------

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                f"best_{metric_name}",
                float(metric_value)
            )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="best_model"
        )