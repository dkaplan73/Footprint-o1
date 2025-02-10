import tensorflow as tf

# Load trained model (without compiling for inference)
model = tf.keras.models.load_model("best_model.h5", compile=False)

# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted TFLite model
with open("map_segmentation_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as map_segmentation_model.tflite.")
