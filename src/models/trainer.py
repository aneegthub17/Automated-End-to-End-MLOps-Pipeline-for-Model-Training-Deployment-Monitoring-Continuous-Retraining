from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


class ModelTrainer:

    def __init__(self):

        self.models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),

            "decision_tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE
            )
        }

    def train(self, X_train, y_train):

        trained_models = {}

        print("\n========== MODEL TRAINING ==========")

        for name, model in self.models.items():

            print(f"\nTraining: {name}")

            model.fit(
                X_train,
                y_train
            )

            trained_models[name] = model

            print(
                f"✓ {name} trained successfully."
            )

        return trained_models