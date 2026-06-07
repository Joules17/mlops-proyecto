FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ARG ENVIRONMENT=dev
ENV ENVIRONMENT=${ENVIRONMENT}

ENV MODEL_PATH=/app/model/model.onnx
ENV BUCKET_NAME=mlops-sentiment-bucket

ARG GCS_MODEL_PATH=gs://mlops-sentiment-bucket/model/model.onnx
RUN mkdir -p /app/model

COPY model/model.onnx /app/model/model.onnx

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
