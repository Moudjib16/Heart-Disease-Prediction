"""Streamlit front-end for the heart-disease risk model.

Loads the fitted preprocessor + tuned XGBoost model (and the clinical
threshold) once, collects patient inputs in the sidebar, and shows a risk
verdict plus a local SHAP explanation for the individual prediction.

Run locally:  streamlit run app/main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load preprocessor, model, threshold and SHAP explainer once per session."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    threshold = (
        json.loads(THRESHOLD_PATH.read_text())["threshold"]
        if THRESHOLD_PATH.exists()
        else 0.5
    )
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, threshold, explainer


def sidebar_inputs() -> dict:
    st.sidebar.header("Patient inputs")
    st.sidebar.caption("Ranges match the training data (4 UCI hospitals).")

    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age (years)", 28, 77, 54)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

    st.sidebar.subheader("Clinical measurements")
    trestbps = st.sidebar.number_input(
        "Resting blood pressure (mm Hg)", 80, 200, 130, step=1
    )
    chol = st.sidebar.number_input(
        "Serum cholesterol (mg/dl)", 85, 603, 240, step=1
    )
    thalch = st.sidebar.slider("Max heart rate achieved", 60, 202, 140)
    oldpeak = st.sidebar.slider(
        "ST depression (oldpeak)", -2.6, 6.2, 0.5, step=0.1
    )
    ca = st.sidebar.select_slider(
        "Major vessels colored (ca)", options=[0, 1, 2, 3], value=0
    )

    st.sidebar.subheader("Clinical findings")
    cp = st.sidebar.selectbox(
        "Chest pain type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    )
    restecg = st.sidebar.selectbox(
        "Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"]
    )
    slope = st.sidebar.selectbox(
        "ST slope", ["upsloping", "flat", "downsloping"]
    )
    thal = st.sidebar.selectbox(
        "Thalassemia", ["normal", "fixed defect", "reversable defect"]
    )
    fbs_label = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    exang_label = st.sidebar.selectbox("Exercise-induced angina", ["No", "Yes"])

    return {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": fbs_label == "Yes",
        "restecg": restecg,
        "thalch": float(thalch),
        "exang": exang_label == "Yes",
        "oldpeak": float(oldpeak),
        "slope": slope,
        "ca": float(ca),
        "thal": thal,
    }


def main() -> None:
    st.title("🫀 Heart Disease Risk Prediction")
    st.caption(
        "XGBoost classifier · explainable with SHAP · educational demo only."
    )

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.error(
            "Model artifacts not found. Train the model first: "
            "`python src/train.py`."
        )
        st.stop()

    preprocessor, model, threshold, explainer = load_artifacts()

    inputs = sidebar_inputs()
    row = pd.DataFrame([inputs])
    row_prep = preprocessor.transform(row)

    proba = float(model.predict_proba(row_prep)[:, 1][0])
    high_risk = proba >= threshold

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Prediction")
        if high_risk:
            st.error(f"⚠️ High risk — {proba:.1%} probability of disease")
        else:
            st.success(f"✅ Low risk — {proba:.1%} probability of disease")
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"Decision threshold: {threshold:.0%} (recall-optimised).")
        st.info("Educational tool — not medical advice.")

    with right:
        st.subheader("Why this prediction?")
        explanation = explainer(row_prep)
        shap.plots.waterfall(explanation[0], show=False, max_display=12)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
