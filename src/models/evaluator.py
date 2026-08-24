from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ModelEvaluator:

    def evaluate(
        self,
        models,
        X_test,
        y_test
    ):

        results = {}

        print("\n========== MODEL EVALUATION ==========")

        for name, model in models.items():

            # Generate class predictions
            predictions = model.predict(
                X_test
            )

            # Generate probability for the positive class
            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            # Convert Yes/No target into 1/0
            y_test_binary = (
                y_test == "Yes"
            ).astype(int)

            # Calculate evaluation metrics
            results[name] = {

                "accuracy": accuracy_score(
                    y_test,
                    predictions
                ),

                "precision": precision_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                ),

                "recall": recall_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                ),

                "f1": f1_score(
                    y_test,
                    predictions,
                    pos_label="Yes",
                    zero_division=0
                ),

                "roc_auc": roc_auc_score(
                    y_test_binary,
                    probabilities
                )
            }

        return results

    def select_best_model(
        self,
        models,
        results,
        metric="f1"
    ):

        best_model_name = max(
            results,
            key=lambda name: results[name][metric]
        )

        best_model = models[
            best_model_name
        ]

        return (
            best_model_name,
            best_model
        )