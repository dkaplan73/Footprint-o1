import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import unet_model
from tensorflow.keras.models import load_model

# Define Dice loss
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Load the model (using custom_objects for dice_loss)
model = load_model('best_model.h5', custom_objects={'dice_loss': dice_loss})
model.compile(optimizer='adam', loss=dice_loss, metrics=["accuracy"])

# For evaluation we use the augmented dataset (change as needed)
data = np.load("dataset_augmented.npz")
X_test, Y_test = data["X_train"], data["Y_train"]

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Visualize one prediction
predictions = model.predict(X_test)
plt.imshow(predictions[0].squeeze(), cmap="gray")
plt.title("Prediction on first test sample")
plt.colorbar()
plt.savefig("evaluation_prediction.png")
print("Evaluation complete. Saved sample prediction as evaluation_prediction.png.")
