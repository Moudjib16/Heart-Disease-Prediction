"""Held-out evaluation, clinical thresholding and SHAP explainability.

Phase 5 pipeline:
  1. Evaluate the saved preprocessor + tuned model on the TEST split only.
  2. Sweep clinical thresholds (a missed case costs far more than a false alarm).
  3. SHAP global explanation (beeswarm) of what drives risk.
  4. Persist metrics (JSON) and figures (PNG) to ``data/processed/``.

Run with ``python src/evaluate.py`` or ``python -m src.evaluate``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import load_clean, split_data
from src.preprocessing import load_preprocessor

MODEL_PATH = Path("models/best_model.pkl")
THRESHOLD_PATH = Path("models/threshold.json")
PROCESSED_DIR = Path("data/processed")
FIG_DIR = PROCESSED_DIR / "figures"

DEFAULT_THRESHOLD = 0.5
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5]
MIN_RECALL_GAIN = 0.02
MAX_PRECISION_DROP = 0.05


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def sweep_thresholds(y_true, y_proba, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        row = {"threshold": t}
        row.update(compute_metrics(y_true, y_pred, y_proba))
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Pick the lowest threshold that buys recall without collapsing precision."""
    base = sweep.loc[DEFAULT_THRESHOLD]
    for threshold, row in sweep.sort_index().iterrows():
        if threshold >= DEFAULT_THRESHOLD:
            continue
        recall_ok = row["recall"] >= base["recall"] + MIN_RECALL_GAIN
        precision_ok = row["precision"] >= base["precision"] - MAX_PRECISION_DROP
        if recall_ok and precision_ok:
            return float(threshold)
    return float(DEFAULT_THRESHOLD)


def save_confusion_matrix(y_true, y_pred, threshold: float) -> None:
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=["no disease", "disease"]
    ).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix (threshold = {threshold})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_roc_curve(y_true, y_proba) -> None:
    RocCurveDisplay.from_predictions(y_true, y_proba, name="XGBoost")
    plt.title("ROC curve (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()


def save_metrics_table(metrics: dict[str, float], threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.axis("off")
    rows = [[k.upper(), f"{v:.3f}"] for k, v in metrics.items()]
    table = ax.table(cellText=rows, colLabels=["Metric", "Score"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title(f"Metrics @ threshold {threshold}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metrics_table.png", dpi=150)
    plt.close()


def save_shap_beeswarm(model, X_test_prep: pd.DataFrame) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_prep)
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    _, X_test, _, y_test = split_data(df)

    preprocessor = load_preprocessor()
    model = joblib.load(MODEL_PATH)
    X_test_prep = preprocessor.transform(X_test)

    y_proba = model.predict_proba(X_test_prep)[:, 1]

    y_pred_default = (y_proba >= DEFAULT_THRESHOLD).astype(int)
    default_metrics = compute_metrics(y_test, y_pred_default, y_proba)

    sweep = sweep_thresholds(y_test, y_proba, SWEEP_THRESHOLDS)
    print("Threshold sweep (test set):")
    print(sweep.round(3).to_string())

    threshold = choose_threshold(sweep)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\nchosen threshold: {threshold}")
    print(classification_report(y_test, y_pred, target_names=["no disease", "disease"]))

    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    summary = {
        "test_set": {
            "n": int(len(y_test)),
            "prevalence": float(y_test.mean()),
        },
        "default_threshold": DEFAULT_THRESHOLD,
        "default_metrics": default_metrics,
        "chosen_threshold": threshold,
        "metrics_at_chosen_threshold": metrics,
        "threshold_sweep": sweep.round(4).to_dict(orient="index"),
    }
    (PROCESSED_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    save_confusion_matrix(y_test, y_pred, threshold)
    save_roc_curve(y_test, y_proba)
    save_metrics_table(metrics, threshold)
    save_shap_beeswarm(model, X_test_prep)

    print(f"\nwrote metrics -> {PROCESSED_DIR / 'metrics.json'}")
    print(f"wrote threshold -> {THRESHOLD_PATH}")
    print(f"wrote figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
