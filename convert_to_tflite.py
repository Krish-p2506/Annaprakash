import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("best_lemon_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save
with open("lemon_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as lemon_model.tflite")
