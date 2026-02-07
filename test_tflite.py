import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="lemon_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

# Class names (IMPORTANT: order must match training)
class_names = ["Anthracnose", "citrus canker", "deficiency", "healthy"]

# Load test image (change path)
img_path = r"D:\Annaprakash\data\dataset\lemon\healthy\sample.jpg"

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Set input
interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])

pred_index = np.argmax(output)
confidence = np.max(output)

print("Predicted:", class_names[pred_index])
print("Confidence:", round(float(confidence)*100, 2), "%")
