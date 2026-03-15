import joblib
import pandas as pd


MODEL_PATH = "models/heart_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict_single(model, patient_data: dict):
    input_df = pd.DataFrame([patient_data])

    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    result = {
        "prediction": int(prediction),
        "risk_probability": round(float(probability), 4),
        "risk_label": "high_risk" if prediction == 1 else "low_risk",
    }
    return result


if __name__ == "__main__":
    model = load_model()

    sample_patient = {
        "age": 63,
        "sex": "Male",
        "cp": "typical angina",
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": True,
        "restecg": "lv hypertrophy",
        "thalch": 150.0,
        "exang": False,
        "oldpeak": 2.3,
    }

    result = predict_single(model, sample_patient)
    print(result)