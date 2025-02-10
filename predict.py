import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the trained model without compiling it (to avoid loss/optimizer configuration issues)
model_path = "best_model.h5"  # Ensure this is the correct model file
model = tf.keras.models.load_model(model_path, compile=False)

# Define the target input size for the model (512x512)
target_size = (512, 512)

# Load the image (ensure that the file exists in the specified folder)
image_path = "evaluation_prediction.png"  # Replace with the actual path to your test image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

# Read the image using OpenCV
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Error: Could not read image at {image_path}. Check file integrity or permissions.")

# Resize the image to match the model input size and normalize it
image_resized = cv2.resize(image, target_size)
image_resized = image_resized.astype("float32") / 255.0

# Add a batch dimension so that the shape becomes (1, 512, 512, 3)
image_resized = np.expand_dims(image_resized, axis=0)

# Run prediction using the model
prediction = model.predict(image_resized).squeeze()

# Threshold the prediction to get a binary mask (adjust threshold as needed)
prediction_binary = (prediction > 0.5).astype(np.uint8)

# Display the original image and the predicted mask side by side
plt.figure(figsize=(10, 5))

# Original image (convert BGR to RGB for display)
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Predicted mask
plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(prediction_binary, cmap="gray")
plt.axis("off")

plt.show()
