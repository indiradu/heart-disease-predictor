import json
import joblib
import pandas as pd


MODEL_PATH = "models/heart_model.pkl"
METRICS_PATH = "models/metrics.json"

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)


def predict_patient(data: dict) -> dict:
    input_df = pd.DataFrame([data])

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
        "disclaimer": "Educational use only. Not a medical diagnosis."
    }


def get_model_info() -> dict:
    return metrics