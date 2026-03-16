from pathlib import Path
import sys
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import predict_local

st.title("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        try:
            results = []
            for _, row in df.iterrows():
                result = predict_local(row.to_dict())
                results.append(result)

            result_df = df.copy()
            result_df["prediction"] = [r["prediction"] for r in results]
            result_df["risk_probability"] = [r["risk_probability"] for r in results]
            result_df["risk_band"] = [r["risk_band"] for r in results]

            st.success("Batch prediction completed")
            st.dataframe(result_df)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")