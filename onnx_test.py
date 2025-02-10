import onnx
import onnxruntime as ort

# Load the ONNX model
onnx_model = onnx.load("best_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Create an inference session
ort_session = ort.InferenceSession("best_model.onnx")
print("ONNX Inference session created successfully.")
