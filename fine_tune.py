import tensorflow as tf
from model import unet_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define Dice loss (same as in training)
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Function to resize and preprocess images and labels to match the model input shape (512x512)
def preprocess_image(img):
    img = tf.image.resize(img, (512, 512))  # Resize image to 512x512
    img = img / 255.0  # Normalize the image
    return img

# Load dataset (original non-augmented data)
data = np.load("dataset.npz")
X_train, Y_train = data["X_train"], data["Y_train"]
X_val, Y_val = data["X_val"], data["Y_val"]

# Resize the training and validation datasets to match the model input size (512x512)
X_train_resized = np.array([preprocess_image(img) for img in X_train])
X_val_resized = np.array([preprocess_image(img) for img in X_val])

# Resize Y_train and Y_val to match the input shape (512, 512, 1)
Y_train_resized = np.array([tf.image.resize(label, (512, 512)) for label in Y_train])
Y_val_resized = np.array([tf.image.resize(label, (512, 512)) for label in Y_val])

# Attempt to load existing model with custom Dice loss
try:
    model = load_model('best_model.h5', custom_objects={'dice_loss': dice_loss})
    print("Model loaded successfully from best_model.h5.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Initializing a new model based on U-Net architecture.")
    input_shape = (512, 512, 3)  # Modify to match the desired input shape (512, 512, 3)
    model = unet_model(input_shape)

# Compile the model (using the same Dice loss for consistency)
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Fine-tune the model on the resized dataset
model.fit(X_train_resized, Y_train_resized, validation_data=(X_val_resized, Y_val_resized), epochs=10, batch_size=8)

# Save the fine-tuned model (overwriting previous best_model.h5)
model.save("best_model.h5")
print("Fine-tuning complete. Model saved as best_model.h5.")
