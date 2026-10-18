"""Model training, cross-validation and tracking for heart-disease risk.

Phase 4 pipeline:
  1. Load + split the cleaned data (stratified, ``random_state=42``).
  2. Cross-validate three baselines on the TRAIN split only.
  3. Tune the strongest candidate (recall-first) with ``RandomizedSearchCV``.
  4. Persist the winner and log params/metrics to MLflow.

Run with ``python src/train.py`` or ``python -m src.train``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier

from src.data_loader import load_clean, split_data
from src.preprocessing import fit_and_save

RANDOM_STATE = 42
EXPERIMENT = "heart-disease-prediction"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_PATH = Path("models/best_model.pkl")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["recall", "roc_auc", "f1"]


def build_models() -> dict[str, object]:
    """The three baseline candidates from the plan."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    }


def evaluate_baselines(models: dict[str, object], X, y) -> pd.DataFrame:
    """Cross-validate each baseline and return mean/std scores per model."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values(
        "recall_mean", ascending=False
    )


def tune_xgboost(X, y) -> RandomizedSearchCV:
    """Randomized search for XGBoost, optimising recall (missed cases cost most)."""
    search = RandomizedSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        param_distributions={
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        n_iter=30,
        cv=CV,
        scoring="recall",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = fit_and_save(X_train)
    X_train_prep = preprocessor.transform(X_train)

    baselines = evaluate_baselines(build_models(), X_train_prep, y_train)
    print("Cross-validated baselines (5-fold, TRAIN only):")
    print(baselines.round(3).to_string())

    search = tune_xgboost(X_train_prep, y_train)
    best_model = search.best_estimator_

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nbest params   : {search.best_params_}")
    print(f"best CV recall: {search.best_score_:.3f}")
    print(f"saved model   -> {MODEL_PATH}")

    cv_scores = cross_validate(
        best_model, X_train_prep, y_train, cv=CV, scoring=SCORING, n_jobs=-1
    )
    metrics = {f"cv_{m}": float(cv_scores[f"test_{m}"].mean()) for m in SCORING}
    metrics["cv_recall_std"] = float(cv_scores["test_recall"].std())

    with mlflow.start_run(run_name="xgb-tuned"):
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

    print("logged tuned run to MLflow:", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
