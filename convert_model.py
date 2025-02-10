import tensorflow as tf

# Load the model without compiling
model = tf.keras.models.load_model("best_model.h5", compile=False)

# Resave the model in HDF5 format (cleaned version)
model.save("best_model_fixed.h5")

# Optionally, save it in TensorFlow's newer format
model.save("best_model.keras")  
