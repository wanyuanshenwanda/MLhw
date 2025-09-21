from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreparedDatasets:
    feature_table: pd.DataFrame
    regression_features: List[str]
    classification_features: List[str]


DATE_COLUMNS = ["created_at", "updated_at", "merged_at", "closed_at"]
DROP_FEATURE_COLUMNS = [
    "title",
    "body",
    "state",
    "merged_at",
    "updated_at",
    "conversation",
    "comments",
    "review_comments",
]
LEAKY_FEATURE_CANDIDATES = {
    "first_comment_at",
    "time_to_first_response_hours",
    "time_to_merge_hours",
    "pr_comment_count",
    "pr_comment_length",
    "reviewer_count",
    "bot_reviewer_count",
    "is_reviewed",
    "last_comment_mention",
}
EMBEDDING_COLUMNS = ["title_embedding", "body_embedding", "comment_embedding"]


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_raw_frames(data_dir: Path) -> Dict[str, pd.DataFrame]:
    frames = {
        "base": _read_excel(data_dir / "PR_info_add_conversation.xlsx"),
        "pr_features": _read_excel(data_dir / "PR_features.xlsx"),
        "author_features": _read_excel(data_dir / "author_features.xlsx"),
        "reviewer_features": _read_excel(data_dir / "reviewer_features.xlsx"),
        "project_features": _read_excel(data_dir / "project_features.xlsx"),
        "comments": _read_excel(data_dir / "PR_comment_info.xlsx"),
    }
    return frames


def _prepare_base_frame(base: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for col in DATE_COLUMNS:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")
    base["created_at"] = pd.to_datetime(base["created_at"], utc=True, errors="coerce")

    base = base.sort_values("created_at").reset_index(drop=True)

    base["title_words"] = base["title"].fillna("").apply(_count_words)
    base["body_words"] = base["body"].fillna("").apply(_count_words)

    comments = comments.copy()
    comments["created_at"] = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")
    first_response = (
        comments.groupby("belong_to_PR")["created_at"].min().rename("first_comment_at")
    )
    base = base.merge(
        first_response,
        left_on="number",
        right_index=True,
        how="left",
    )

    base["time_to_first_response_hours"] = (
        (base["first_comment_at"] - base["created_at"]).dt.total_seconds() / 3600.0
    )

    base["time_to_close_hours"] = (
        (base["closed_at"] - base["created_at"]).dt.total_seconds() / 3600.0
    )
    base["time_to_close_days"] = base["time_to_close_hours"] / 24.0

    base["time_to_merge_hours"] = (
        (base["merged_at"] - base["created_at"]).dt.total_seconds() / 3600.0
    )

    base["prev_prs"] = base.groupby("author").cumcount()

    return base


def _count_words(text: str) -> int:
    tokens = str(text).strip().split()
    return len(tokens)


def _prepare_pr_features(pr_features: pd.DataFrame) -> pd.DataFrame:
    pr_features = pr_features.drop(columns=[c for c in EMBEDDING_COLUMNS if c in pr_features], errors="ignore")
    rename_map = {
        "directory_num": "directories",
        "language_num": "language_types",
        "file_type": "file_types",
        "segs_updated": "segs_changed",
        "files_updated": "files_changed",
        "title_length": "title_chars",
        "body_length": "body_chars",
        "comment_num": "pr_comment_count",
        "comment_length": "pr_comment_length",
        "reviewer_num": "reviewer_count",
        "bot_reviewer_num": "bot_reviewer_count",
    }
    pr_features = pr_features.rename(columns=rename_map)
    return pr_features


def _prefix_columns(frame: pd.DataFrame, prefix: str, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    rename_map = {
        col: f"{prefix}{col}" for col in frame.columns if col not in exclude
    }
    return frame.rename(columns=rename_map)


def build_feature_table(data_dir: Path) -> PreparedDatasets:
    frames = load_raw_frames(data_dir)
    base = _prepare_base_frame(frames["base"], frames["comments"])
    pr_features = _prepare_pr_features(frames["pr_features"])

    author_features = frames["author_features"].copy()
    author_features = _prefix_columns(author_features, "author_", exclude=["number"])
    author_features = author_features.rename(columns={"author_name": "author_username"})

    reviewer_features = frames["reviewer_features"].copy()
    reviewer_features = _prefix_columns(reviewer_features, "reviewer_", exclude=["number"])
    reviewer_features = reviewer_features.rename(columns={"reviewer_name": "reviewer_primary"})

    project_features = frames["project_features"].copy()
    project_features = _prefix_columns(project_features, "project_", exclude=["number"])

    merged = base.merge(pr_features, on="number", how="left")
    merged = merged.merge(author_features, on="number", how="left")
    merged = merged.merge(reviewer_features, on="number", how="left")
    merged = merged.merge(project_features, on="number", how="left")

    merged = merged.drop(columns=[c for c in DROP_FEATURE_COLUMNS if c in merged], errors="ignore")
    merged = merged.drop(columns=[c for c in LEAKY_FEATURE_CANDIDATES if c in merged], errors="ignore")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    boolean_cols = merged.select_dtypes(include=["bool"]).columns.tolist()
    for col in boolean_cols:
        merged[col] = merged[col].astype(int)

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    regression_features = [
        col
        for col in numeric_cols
        if col
        not in {
            "number",
            "merged",
            "time_to_close_hours",
            "time_to_close_days",
        }
    ]

    classification_features = [
        col
        for col in numeric_cols
        if col
        not in {
            "number",
            "merged",
            "time_to_close_hours",
            "time_to_close_days",
            "time_to_merge_hours",
        }
    ]

    merged = merged.sort_values("created_at").reset_index(drop=True)

    return PreparedDatasets(
        feature_table=merged,
        regression_features=regression_features,
        classification_features=classification_features,
    )


def split_by_time(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = df.dropna(subset=[target_column]).copy()
    mask = df["created_at"] < pd.to_datetime(split_date, utc=True)
    train_df = df[mask]
    test_df = df[~mask]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    return X_train, y_train, X_test, y_test


def fill_missing_within_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    filled = df.copy()
    for col in feature_columns:
        if filled[col].isna().any():
            filled[col] = filled[col].fillna(filled[col].median())
    return filled
