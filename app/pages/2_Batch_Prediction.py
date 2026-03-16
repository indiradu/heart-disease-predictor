import pandas as pd
import streamlit as st
from app.utils import predict_api

st.title("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        results = []

        try:
            for _, row in df.iterrows():
                payload = row.to_dict()
                result = predict_api(payload)
                results.append(result)

            result_df = df.copy()
            result_df["prediction"] = [r["prediction"] for r in results]
            result_df["risk_probability"] = [r["risk_probability"] for r in results]
            result_df["risk_band"] = [r["risk_band"] for r in results]

            st.success("Batch prediction completed")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")