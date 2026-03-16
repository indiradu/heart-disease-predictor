from fastapi import FastAPI
from api.schemas import PatientData
from api.services import predict_patient, get_model_info

app = FastAPI(
    title="Heart Disease Predictor API",
    description="Educational API for heart disease risk prediction",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "Heart Disease Predictor API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return get_model_info()


@app.post("/predict")
def predict(data: PatientData):
    return predict_patient(data.model_dump())