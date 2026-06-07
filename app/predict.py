import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.onnx")
TOKENIZER_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
session = ort.InferenceSession(MODEL_PATH)

LABELS = ["non-toxic", "toxic"]


def predict(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0][0]

    probs = np.exp(logits) / np.sum(np.exp(logits))
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "text": text,
        "label": LABELS[predicted_class],
        "toxic": bool(predicted_class == 1),
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "non-toxic": round(float(probs[0]) * 100, 2),
            "toxic": round(float(probs[1]) * 100, 2),
        },
    }
