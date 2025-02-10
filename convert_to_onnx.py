import tensorflow as tf
import tf2onnx

# Load the model without compiling to avoid loss configuration issues.
model = tf.keras.models.load_model("best_model.h5", compile=False)

# Set a fixed batch size in the input signature. This helps with shape inference.
# For example, if the model expects input shape (512,512,3), then with batch size 1, the shape is (1,512,512,3).
input_signature = [tf.TensorSpec(shape=(1,) + model.input_shape[1:], dtype=tf.float32)]

# Convert the Keras model to ONNX format (using opset 13)
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13, input_signature=input_signature)

# Save the ONNX model to file.
with open("best_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Conversion to ONNX complete. Saved as best_model.onnx.")
