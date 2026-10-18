"""Single source of truth for the heart-disease preprocessing pipeline.

Both training (``src/train.py``) and the Streamlit app import from here so the
column lists, imputation and encoding rules are defined exactly once and can
never drift between fit time and predict time.

Design notes (justified in ``notebooks/01_eda.ipynb``):
  * ``ca`` (major vessels, 0-3) is an ordinal count, so it stays **numeric**.
  * ``add_indicator=True`` flags which values were imputed; ``ca`` is ~66%
    missing, so "was this measured?" can itself be informative.
  * One-hot output is dense (``sparse_output=False``) and pandas-labelled
    (``set_output(transform="pandas")``) so SHAP plots and the app get
    readable feature names; the dataset is far too small for sparse to help.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = Path("models/preprocessor.pkl")

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh, unfitted preprocessing ``ColumnTransformer``."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    ).set_output(transform="pandas")


def fit_and_save(
    X_train: pd.DataFrame, path: str | Path = MODEL_PATH
) -> ColumnTransformer:
    """Fit the preprocessor on TRAIN data only and persist it to ``path``."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)

    return preprocessor


def load_preprocessor(path: str | Path = MODEL_PATH) -> ColumnTransformer:
    """Load a previously fitted preprocessor."""
    return joblib.load(path)


if __name__ == "__main__":
    from src.data_loader import load_clean, split_data

    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = fit_and_save(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print(f"saved preprocessor -> {MODEL_PATH}")
    print(f"train transformed : {X_train_prep.shape}")
    print(f"test transformed  : {X_test_prep.shape}")
    print(f"feature names     : {len(preprocessor.get_feature_names_out())}")
    print(f"NaN in train      : {int(X_train_prep.isna().sum().sum())}")
    print(f"NaN in test       : {int(X_test_prep.isna().sum().sum())}")
