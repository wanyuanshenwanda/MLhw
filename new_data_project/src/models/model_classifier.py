from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class MultiProjectClassifier:
    def __init__(self, model_type="LogisticRegression", **kwargs):
        self.model_type = model_type
        self.model_kwargs = kwargs
        self.models = []
        self.metrics = []

    def _get_classifier(self):
        if self.model_type == "LogisticRegression":
            return LogisticRegression(**self.model_kwargs)
        elif self.model_type == "RandomForest":
            return RandomForestClassifier(**self.model_kwargs)
        elif self.model_type == "GradientBoosting":
            return GradientBoostingClassifier(**self.model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train_list, y_train_list, X_test_list=None, y_test_list=None):
        self.models = []
        self.metrics = []

        for i, X_train in enumerate(X_train_list):
            y_train = y_train_list[i]
            X_test = X_test_list[i] if X_test_list else None
            y_test = y_test_list[i] if y_test_list else None

            model = self._get_classifier()
            model.fit(X_train, y_train)
            self.models.append(model)

            metric = {}
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                metric = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "F1": f1_score(y_test, y_pred, average="macro", zero_division=0)
                }
                print(f"Dataset {i} metrics:", metric)
            self.metrics.append(metric)

    # def predict(self, X_list):
    #     preds = []
    #     for i, X in enumerate(X_list):
    #         preds.append(self.models[i].predict(X))
    #     return preds
