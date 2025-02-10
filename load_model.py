import tensorflow as tf

model = tf.keras.models.load_model("best_model.h5", compile=False)
model.summary()  # Check if the model loads correctly

