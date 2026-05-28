from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

MODEL_FILE = 'keras_model.h5'
LABELS_FILE = 'labels.txt'

app = FastAPI(title='Clasificador de Aves API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def load_labels():
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"No se encuentra '{LABELS_FILE}'")

    labels = []
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            labels.append(parts[1].strip() if len(parts) > 1 else parts[0].strip())
    return labels


def load_keras_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No se encuentra '{MODEL_FILE}'")

    return load_model(MODEL_FILE, compile=False)


try:
    model = load_keras_model()
    class_labels = load_labels()
except Exception as e:
    model = None
    class_labels = []
    print(f'Error cargando modelo o etiquetas: {e}')


@app.get('/api/health')
def health_check():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/api/labels')
def get_labels():
    return {'labels': class_labels}


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail='Modelo no cargado')

    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail='El archivo no es una imagen válida')

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        normalized_image = (image_array / 127.5) - 1.0
        input_tensor = np.expand_dims(normalized_image, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0]
        top_index = int(np.argmax(prediction))
        top_label = class_labels[top_index] if top_index < len(class_labels) else f'Clase {top_index}'
        confidence = float(prediction[top_index])

        results = [
            {
                'class': class_labels[i] if i < len(class_labels) else f'Clase {i}',
                'probability': float(prediction[i]),
                'percentage': f'{prediction[i] * 100:.2f}'
            }
            for i in range(len(prediction))
        ]

        results.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'className': top_label,
            'confidence': confidence * 100,
            'probability': confidence,
            'allPredictions': results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error al procesar la imagen: {exc}')
