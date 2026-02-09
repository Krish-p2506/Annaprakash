from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Annaprakash AI API")

model = tf.keras.models.load_model("best_lemon_model.h5")

classes = [
    "Anthracnose",
    "Citrus Canker",
    "Deficiency",
    "Healthy"
]

IMG_SIZE = 224


def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    x = preprocess(img)

    pred = model.predict(x)

    idx = int(np.argmax(pred))
    conf = float(np.max(pred))

    return {
        "disease": classes[idx],
        "confidence": round(conf * 100, 2)
    }
