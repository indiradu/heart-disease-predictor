from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

_model = None
_metrics = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def load_metrics():
    global _metrics
    if _metrics is None:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _metrics = json.load(f)
    return _metrics


def predict_local(payload: dict) -> dict:
    model = load_model()
    input_df = pd.DataFrame([payload])

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])

    if probability < 0.30:
        risk_band = "low"
    elif probability < 0.70:
        risk_band = "moderate"
    else:
        risk_band = "high"

    return {
        "prediction": prediction,
        "risk_probability": round(probability, 4),
        "risk_label": "high_risk" if prediction == 1 else "low_risk",
        "risk_band": risk_band,
        "disclaimer": "Educational use only. Not a medical diagnosis.",
    }


def get_model_info() -> dict:
    return load_metrics()