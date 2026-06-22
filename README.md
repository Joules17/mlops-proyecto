# MLOPS Proyecto Final - Clasificador de comentarios toxicos 

Sistema de despliegue automático para un modelo de clasificación de comentarios tóxicos en español e inglés, usando ONNX Runtime, FastAPI y Google Cloud Platform.

## Arquitectura

```
GitHub Push (dev/prod)
        ↓
GitHub Actions Pipeline
    ├── test: Descarga modelo + datos → corre pruebas unitarias
    └── build: Construye Docker → sube a Artifact Registry → despliega en Cloud Run
        ↓
Cloud Run (dev endpoint / prod endpoint)
        ↓
Predicciones guardadas en GCS (predicciones_dev.txt / predicciones_prod.txt)
```

## Estructura del repositorio

```
mlops-proyecto/
├── app/
│   ├── main.py          # FastAPI app
│   └── predict.py       # Lógica de inferencia ONNX
├── tests/
│   └── test_model.py    # Pruebas unitarias
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        ├── dev.yml      # Pipeline para rama dev
        └── prod.yml     # Pipeline para rama prod
```

## Modelo

- **Modelo:** `onnx-community/bert-multilingual-toxicity-classifier-ONNX` 
- **Formato:** ONNX
- **Almacenamiento:** Google Cloud Storage (`gs://mlops-sentiment-bucket/model/model.onnx`)
- **Tarea:** Clasificación binaria de toxicidad en comentarios (tóxico: 1 / no tóxico: 0)
- **Idiomas:** Multilingue - Pruebas funcionales en español

## Modelo alterno  
- **Modelo:** `Xenova/distilbert-base-multilingual-cased-sentiments-student`  

## Pruebas unitarias

El pipeline corre dos pruebas antes de desplegar:

1. **test_model_returns_valid_output:** Verifica que el modelo retorna predicciones válidas (clase 0 o 1, confianza entre 0 y 1)
2. **test_model_accuracy_above_threshold:** Verifica que la accuracy del modelo en los datos de prueba sea mayor al 65%

## Endpoints

| Ambiente | URL |
|---|---|
| DEV | `https://toxic-classifier-dev-xxxx.run.app` |
| PROD | `https://toxic-classifier-prod-xxxx.run.app` |

## API

### POST /predict
```json
{
  "text": "Tu comentario aquí"
}
```

Respuesta:
```json
{
  "text": "Tu comentario aquí",
  "label": "toxic",
  "toxic": true,
  "confidence": 95.3,
  "probabilities": {
    "non-toxic": 4.7,
    "toxic": 95.3
  },
  "environment": "dev",
  "timestamp": "2024-01-01T00:00:00"
}
```

## Infraestructura GCP

| Recurso | Nombre |
|---|---|
| Proyecto | `mlops-sentiment` |
| Bucket | `mlops-sentiment-bucket` |
| Artifact Registry | `toxic-classifier` |
| Cloud Run DEV | `toxic-classifier-dev` |
| Cloud Run PROD | `toxic-classifier-prod` |

## Secrets de GitHub

| Secret | Descripción |
|---|---|
| `GCP_PROJECT_ID` | ID del proyecto en GCP |
| `GCP_SA_KEY` | JSON key de la Service Account |

## Monitoreo de predicciones

Cada predicción se guarda en GCS:
- DEV: `gs://mlops-sentiment-bucket/predictions/predicciones_dev.txt`
- PROD: `gs://mlops-sentiment-bucket/predictions/predicciones_prod.txt`

Formato:
```
2024-01-01T00:00:00 | text: Eres un idiota | label: toxic | confidence: 95.3%
```
