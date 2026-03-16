import requests

API_URL = "http://127.0.0.1:8000"


def predict_api(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict:
    response = requests.get(f"{API_URL}/model-info", timeout=30)
    response.raise_for_status()
    return response.json()