from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]


def regression_models_definition() -> Dict[str, Pipeline]:
    scaler = ColumnTransformer([
        ("num", StandardScaler(), slice(0, None)),
    ])
    return {
        "LinearRegression": Pipeline([
            ("scaler", scaler),
            ("model", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "GradientBoostingLogTarget": TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(random_state=42),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def classification_models_definition() -> Dict[str, Pipeline]:
    scaler = ColumnTransformer([
        ("num", StandardScaler(), slice(0, None)),
    ])
    return {
        "LogisticRegression": Pipeline([
            ("scaler", scaler),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    results: List[Dict[str, float]] = []
    fitted_models: Dict[str, Pipeline] = {}
    models = regression_models_definition()
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        results.append({"model": name, **metrics})
        fitted_models[name] = model
    return pd.DataFrame(results), fitted_models


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    results: List[Dict[str, float]] = []
    fitted_models: Dict[str, Pipeline] = {}
    models = classification_models_definition()
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_classification(y_test, preds)
        results.append({"model": name, **metrics})
        fitted_models[name] = model
    return pd.DataFrame(results), fitted_models


def make_ablation(
    model_factory: Callable[[], Pipeline],
    feature_groups: Dict[str, List[str]],
    feature_columns: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
) -> pd.DataFrame:
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    baseline_model = model_factory()
    baseline_model.fit(X_train[feature_columns], y_train)
    baseline_preds = baseline_model.predict(X_test[feature_columns])

    if task == "regression":
        baseline_metrics = evaluate_regression(y_test, baseline_preds)
    else:
        baseline_metrics = evaluate_classification(y_test, baseline_preds)

    rows = [{"scenario": "baseline", **baseline_metrics}]

    for group, cols in feature_groups.items():
        cols_to_remove = [c for c in cols if c in feature_columns]
        if not cols_to_remove:
            continue
        remaining = [c for c in feature_columns if c not in cols_to_remove]
        if not remaining:
            continue
        model = model_factory()
        model.fit(X_train[remaining], y_train)
        preds = model.predict(X_test[remaining])
        if task == "regression":
            metrics = evaluate_regression(y_test, preds)
        else:
            metrics = evaluate_classification(y_test, preds)
        rows.append({"scenario": f"remove_{group}", **metrics})

    return pd.DataFrame(rows)
