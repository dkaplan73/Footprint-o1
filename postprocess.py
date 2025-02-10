import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model without compiling (for inference)
model = tf.keras.models.load_model("best_model.h5", compile=False)

def postprocess_mask(mask):
    """
    Applies morphological operations to clean segmentation masks.
    """
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise and fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def predict_and_clean(image_path, output_path):
    """
    Predicts traversable areas and refines the output for a given image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Resize the image to the expected input size of 512x512
    image_resized = cv2.resize(image, (512, 512)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    prediction = model.predict(image_resized)
    prediction = prediction.squeeze()
    print(f"Processing {image_path} - Prediction min/max: {prediction.min()}/{prediction.max()}")
    
    # Threshold the prediction to obtain a binary mask
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    
    # Save raw prediction for debugging
    raw_output_path = os.path.join(output_path, "raw_" + os.path.basename(image_path))
    cv2.imwrite(raw_output_path, prediction)
    
    # Post-process prediction
    cleaned_mask = postprocess_mask(prediction)
    
    # Save processed mask
    output_mask_path = os.path.join(output_path, "mask_" + os.path.basename(image_path))
    cv2.imwrite(output_mask_path, cleaned_mask)
    print(f"Saved processed mask: {output_mask_path}")

def process_images(input_folder="images", output_folder="output_masks"):
    """
    Processes all images in the input folder and saves masks in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(input_folder, filename)
            predict_and_clean(image_path, output_folder)

if __name__ == "__main__":
    process_images()
