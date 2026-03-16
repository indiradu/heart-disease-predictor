from pydantic import BaseModel, Field


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: str
    cp: str
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: bool
    restecg: str
    thalch: float = Field(..., ge=50, le=250)
    exang: bool
    oldpeak: float = Field(..., ge=0, le=10)