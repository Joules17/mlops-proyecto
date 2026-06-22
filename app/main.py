from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.predict import predict
from google.cloud import storage
import os
from datetime import datetime

app = FastAPI(
    title="Toxic Comment Classifier",
    description="API para clasificación de comentarios tóxicos en español e inglés",
    version="1.0.0",
)

BUCKET_NAME = os.getenv("BUCKET_NAME", "mlops-sentiment-bucket")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
PREDICTIONS_FILE = f"predictions/predicciones_{ENVIRONMENT}.txt"


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    label: str
    toxic: bool
    confidence: float
    probabilities: dict
    environment: str
    timestamp: str


def log_prediction(result: dict):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(PREDICTIONS_FILE)

        timestamp = datetime.utcnow().isoformat()
        new_line = f"{timestamp} | text: {result['text']} | label: {result['label']} | confidence: {result['confidence']}%\n"

        try:
            existing = blob.download_as_text()
        except Exception:
            existing = ""

        blob.upload_from_string(existing + new_line, content_type="text/plain")
    except Exception as e:
        print(f"Error logging prediction: {e}")


@app.get("/")
def root():
    return {
        "message": "Toxic Comment Classifier API v.2.0 Prueba",
        "environment": ENVIRONMENT,
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "exitoso", "environment": ENVIRONMENT}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    result = predict(request.text)
    log_prediction(result)

    return {
        **result,
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
    }
