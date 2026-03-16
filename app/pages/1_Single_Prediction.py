from pathlib import Path
import sys
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import predict_local

st.title("Single Patient Prediction")
st.write("Enter patient information to estimate heart disease risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        )
        trestbps = st.number_input("Resting Blood Pressure", min_value=50.0, max_value=300.0, value=130.0)
        chol = st.number_input("Cholesterol", min_value=50.0, max_value=700.0, value=220.0)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
        restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Max Heart Rate", min_value=50.0, max_value=250.0, value=150.0)
        exang = st.selectbox("Exercise Induced Angina", [True, False])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
    }

    try:
        result = predict_local(payload)
        st.subheader("Prediction Result")
        st.metric("Risk Probability", f"{result['risk_probability'] * 100:.1f}%")
        st.metric("Risk Band", result["risk_band"].title())

        if result["risk_band"] == "high":
            st.error("High predicted risk")
        elif result["risk_band"] == "moderate":
            st.warning("Moderate predicted risk")
        else:
            st.success("Low predicted risk")

        st.caption(result["disclaimer"])
    except Exception as e:
        st.error(f"Prediction failed: {e}")