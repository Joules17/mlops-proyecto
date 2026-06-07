import pytest
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.onnx")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "data/test_data_esp.csv")
TOKENIZER_NAME = "bert-base-multilingual-cased"
ACCURACY_THRESHOLD = 0.65


@pytest.fixture(scope="module")
def model_session():
    """Carga el modelo ONNX una sola vez para todos los tests"""
    assert os.path.exists(MODEL_PATH), f"Modelo no encontrado en {MODEL_PATH}"
    session = ort.InferenceSession(MODEL_PATH)
    return session


@pytest.fixture(scope="module")
def tokenizer():
    """Carga el tokenizer una sola vez para todos los tests"""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture(scope="module")
def test_data():
    """Carga los datos de prueba"""
    assert os.path.exists(TEST_DATA_PATH), f"Datos no encontrados en {TEST_DATA_PATH}"
    df = pd.read_csv(TEST_DATA_PATH, sep=";")
    return df


def run_inference(session, tokenizer, text: str):
    """Helper para correr inferencia"""
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
    return int(np.argmax(probs)), float(np.max(probs))


# PRUEBA 1: El modelo responde con datos de entrada definidos
def test_model_returns_valid_output(model_session, tokenizer):
    """Prueba que el modelo retorna una predicción válida dado un texto de entrada"""
    test_texts = [
        "Eres un idiota sin cerebro",
        "Excelente trabajo, muy bien explicado",
        "Ojalá te mueras miserable",
        "Qué hermoso día para aprender",
    ]

    for text in test_texts:
        predicted_class, confidence = run_inference(model_session, tokenizer, text)

        # Verificar que la clase es 0 o 1
        assert predicted_class in [0, 1], (
            f"La clase predicha debe ser 0 o 1, se obtuvo: {predicted_class}"
        )

        # Verificar que la confianza está entre 0 y 1
        assert 0.0 <= confidence <= 1.0, (
            f"La confianza debe estar entre 0 y 1, se obtuvo: {confidence}"
        )

        print(f"texto: '{text[:40]}...' → clase: {predicted_class}, confianza: {confidence:.2f}")


# PRUEBA 2: La métrica no baja del umbral definido
def test_model_accuracy_above_threshold(model_session, tokenizer, test_data):
    correct = 0
    total = len(test_data)

    for _, row in test_data.iterrows():
        text = str(row["text"])
        true_label = int(row["toxic"])
        predicted_class, _ = run_inference(model_session, tokenizer, text)

        if predicted_class == true_label:
            correct += 1

    accuracy = correct / total
    print(f"\n Accuracy del modelo: {accuracy:.2%} ({correct}/{total})")
    print(f"Umbral mínimo requerido: {ACCURACY_THRESHOLD:.2%}")

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"La accuracy {accuracy:.2%} es menor al umbral mínimo requerido {ACCURACY_THRESHOLD:.2%}"
    )
