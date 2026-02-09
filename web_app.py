from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model("best_lemon_model.h5")

class_names = ["Anthracnose", "citrus canker", "deficiency", "healthy"]

IMG_SIZE = 224

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:Arial; text-align:center; margin-top:50px;">
        <h2>Annaprakash - Lemon Disease Detection</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    conf = np.max(pred)

    return f"""
    <html>
    <body style="font-family:Arial; text-align:center; margin-top:50px;">
        <h2>Result</h2>
        <p><b>Disease:</b> {class_names[idx]}</p>
        <p><b>Confidence:</b> {conf*100:.2f}%</p>
        <a href="/">Test Another</a>
    </body>
    </html>
    """
