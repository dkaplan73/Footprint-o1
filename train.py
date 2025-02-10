import tensorflow as tf
from model import unet_model
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("dataset.npz")
X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]

# Build model
model = unet_model()

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=8)

# Save model
model.save("map_segmentation_model.h5")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

