from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .data_preparation import build_feature_table, split_by_time
from .modeling import (
    make_ablation,
    train_classification_models,
    train_regression_models,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

SPLIT_DATE = "2021-06-01"


def ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)


def fill_missing_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: List[str],
) -> (pd.DataFrame, pd.DataFrame):
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    for col in feature_columns:
        median = X_train_filled[col].median()
        if pd.isna(median):
            median = 0.0
        X_train_filled[col] = X_train_filled[col].fillna(median)
        X_test_filled[col] = X_test_filled[col].fillna(median)
    return X_train_filled, X_test_filled


def configure_chinese_fonts() -> None:
    """Prioritize Chinese-capable fonts for matplotlib output."""
    font_candidates = [
        "SimHei",
        "Microsoft Ya Hei",
        "PingFang SC",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]
    existing = plt.rcParams.get("font.sans-serif", [])
    merged = font_candidates + [font for font in existing if font not in font_candidates]
    plt.rcParams["font.sans-serif"] = merged
    plt.rcParams["axes.unicode_minus"] = False

def build_feature_groups(feature_columns: List[str]) -> Dict[str, List[str]]:
    feature_set = set(feature_columns)
    textual = [
        col
        for col in feature_columns
        if col.startswith("has_")
        or col in {
            "title_words",
            "title_chars",
            "title_readability",
            "body_words",
            "body_chars",
            "body_readability",
        }
    ]
    structure = [
        col
        for col in ["directories", "language_types", "file_types"]
        if col in feature_set
    ]
    churn = [
        col
        for col in [
            "lines_added",
            "lines_deleted",
            "segs_added",
            "segs_deleted",
            "segs_changed",
            "files_added",
            "files_deleted",
            "files_changed",
            "modify_proportion",
            "modify_entropy",
            "test_churn",
            "non_test_churn",
            "commits",
            "additions",
            "deletions",
            "changed_files",
        ]
        if col in feature_set
    ]
    author_profile = [
        col for col in feature_columns if col.startswith("author_") or col == "prev_prs"
    ]
    project_profile = [col for col in feature_columns if col.startswith("project_")]
    reviewer_network = [col for col in feature_columns if col.startswith("reviewer_")]
    groups = {
        "textual": textual,
        "structure": structure,
        "code_churn": churn,
        "author_profile": author_profile,
        "project_profile": project_profile,
        "reviewer_network": reviewer_network,
    }
    return {k: v for k, v in groups.items() if v}


def plot_metrics(df: pd.DataFrame, metrics: List[str], title: str, output_path: Path) -> None:
    melted = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ablation(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="scenario", y=metric, color="#4C72B0")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_date: str = SPLIT_DATE,
) -> Dict[str, pd.DataFrame]:
    ensure_output_dirs(output_dir)
    prepared = build_feature_table(data_dir)
    feature_table = prepared.feature_table
    feature_table.to_csv(output_dir / "feature_table.csv", index=False)

    configure_chinese_fonts()
    sns.set_theme(style="whitegrid", context="talk")

    results: Dict[str, pd.DataFrame] = {}

    # Regression task: Time-to-close prediction
    regression_df = feature_table.copy()
    reg_features = prepared.regression_features
    X_train_reg, y_train_reg, X_test_reg, y_test_reg = split_by_time(
        regression_df,
        reg_features,
        "time_to_close_hours",
        split_date,
    )
    X_train_reg, X_test_reg = fill_missing_train_test(X_train_reg, X_test_reg, reg_features)

    reg_metrics, reg_models = train_regression_models(
        X_train_reg,
        y_train_reg,
        X_test_reg,
        y_test_reg,
    )
    reg_metrics.to_csv(output_dir / "regression_metrics.csv", index=False)
    plot_metrics(
        reg_metrics,
        metrics=["mae", "rmse", "r2"],
        title="Regression Metrics (Test Set)",
        output_path=output_dir / "figures" / "regression_metrics.png",
    )
    results["regression_metrics"] = reg_metrics

    reg_groups = build_feature_groups(reg_features)
    reg_ablation = make_ablation(
        model_factory=lambda: TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(random_state=42),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        feature_groups=reg_groups,
        feature_columns=reg_features,
        X_train=X_train_reg,
        y_train=y_train_reg,
        X_test=X_test_reg,
        y_test=y_test_reg,
        task="regression",
    )
    reg_ablation.to_csv(output_dir / "regression_feature_ablation.csv", index=False)
    plot_ablation(
        reg_ablation,
        metric="mae",
        title="Regression Feature Ablation (MAE)",
        output_path=output_dir / "figures" / "regression_ablation_mae.png",
    )
    results["regression_ablation"] = reg_ablation

    classification_df = feature_table[feature_table["closed_at"].notna()].copy()
    cls_features = prepared.classification_features
    X_train_cls, y_train_cls, X_test_cls, y_test_cls = split_by_time(
        classification_df,
        cls_features,
        "merged",
        split_date,
    )
    X_train_cls, X_test_cls = fill_missing_train_test(X_train_cls, X_test_cls, cls_features)

    y_train_cls = y_train_cls.astype(int)
    y_test_cls = y_test_cls.astype(int)

    cls_metrics, cls_models = train_classification_models(
        X_train_cls,
        y_train_cls,
        X_test_cls,
        y_test_cls,
    )
    cls_metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
    plot_metrics(
        cls_metrics,
        metrics=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        title="Classification Metrics (Test Set)",
        output_path=output_dir / "figures" / "classification_metrics.png",
    )
    results["classification_metrics"] = cls_metrics

    cls_groups = build_feature_groups(cls_features)
    cls_ablation = make_ablation(
        model_factory=lambda: GradientBoostingClassifier(random_state=42),
        feature_groups=cls_groups,
        feature_columns=cls_features,
        X_train=X_train_cls,
        y_train=y_train_cls,
        X_test=X_test_cls,
        y_test=y_test_cls,
        task="classification",
    )
    cls_ablation.to_csv(output_dir / "classification_feature_ablation.csv", index=False)
    plot_ablation(
        cls_ablation,
        metric="f1_macro",
        title="Classification Feature Ablation (F1 Macro)",
        output_path=output_dir / "figures" / "classification_ablation_f1.png",
    )
    results["classification_ablation"] = cls_ablation

    return results


if __name__ == "__main__":
    run_pipeline()
