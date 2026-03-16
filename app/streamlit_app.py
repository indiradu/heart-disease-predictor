from pathlib import Path
import sys
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
)

st.title("Heart Disease Predictor")
st.write("Educational AI application for estimating heart disease risk.")
st.info("This tool is for educational use only and is not a medical diagnosis.")

pg = st.navigation(
    [
        st.Page(str(BASE_DIR / "pages" / "1_Single_Prediction.py"), title="Single Prediction"),
        st.Page(str(BASE_DIR / "pages" / "2_Batch_Prediction.py"), title="Batch Prediction"),
        st.Page(str(BASE_DIR / "pages" / "3_Model_Performance.py"), title="Model Performance"),
        st.Page(str(BASE_DIR / "pages" / "4_About.py"), title="About"),
    ]
)

pg.run()