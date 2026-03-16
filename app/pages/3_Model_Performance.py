from pathlib import Path
import sys
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import get_model_info

st.title("Model Performance")

try:
    info = get_model_info()
    st.subheader("Best Model")
    st.write(info["best_model"])

    metrics = info["best_model_metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", metrics["accuracy"])
    col2.metric("F1 Score", metrics["f1_score"])
    col3.metric("ROC AUC", metrics["roc_auc"])

    st.subheader("All Models")
    st.json(info["all_models"])
except Exception as e:
    st.error(f"Could not load model metrics: {e}")