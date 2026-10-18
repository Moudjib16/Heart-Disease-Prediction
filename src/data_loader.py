"""Loading and cleaning for the UCI Heart Disease dataset.

Single documented cleaning rules. Both the EDA notebook
and the later production pipeline import from here so the rules are never
duplicated or allowed to drift.

Dataset: Kaggle "UCI Heart Disease" redo (redwankarimsony/heart-disease-data),
920 patients across 4 hospitals. Raw file: data/raw/heart_disease_uci.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/heart_disease_uci.csv")

ID_COLUMNS = ["id", "dataset"]

IMPLAUSIBLE_ZERO = ["chol", "trestbps"]

TARGET_SOURCE = "num"

TARGET_COL = "target"

RANDOM_STATE = 42


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV without modification."""
    return pd.read_csv(path)


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Phase 2 cleaning rules and return features plus ``target``.

    Rules (applied before any train/test split):
      * Drop ``id`` (row index) and ``dataset`` (hospital source, which
        strongly confounds the target: disease rate ranges 36%-94% by site).
      * Replace clinically implausible zeros in ``chol`` and ``trestbps``
        with NaN so they are imputed rather than treated as real measurements.
      * Binarize the multi-class ``num`` (0-4) into ``target = num > 0``.
      * Drop exact duplicate rows (which only surface once ``id`` is removed).
      * Keep missing values in ``ca`` / ``thal``; they are imputed downstream.
    """
    out = df.copy()

    out = out.drop(columns=[c for c in ID_COLUMNS if c in out.columns])

    out[IMPLAUSIBLE_ZERO] = out[IMPLAUSIBLE_ZERO].replace(0, pd.NA)

    out[TARGET_COL] = (out[TARGET_SOURCE] > 0).astype(int)
    out = out.drop(columns=[TARGET_SOURCE])

    out = out.drop_duplicates().reset_index(drop=True)

    return out


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features ``X`` and binary ``y``."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a stratified train/test split of a cleaned frame.

    Stratification preserves the disease prevalence in both folds, which
    matters on a small (~918-row) dataset; ``random_state`` makes the split
    reproducible. Returns ``X_train, X_test, y_train, y_test``.
    """
    X, y = get_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def load_clean(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Convenience wrapper: load the raw file and return the cleaned frame."""
    return clean_heart_data(load_raw(path))
