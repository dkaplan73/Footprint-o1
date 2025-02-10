import cv2
import glob
import os
import numpy as np

# Define paths to your images and annotations folders.
image_folder = 'images'
annotation_folder = 'images'

def load_data(image_folder=image_folder, annotation_folder=annotation_folder):
    """
    Loads images and corresponding annotation masks.
    All images and masks are resized to 512x512.
    Annotations are thresholded to binary (0/1).
    
    Returns:
        images: NumPy array of images.
        annotations: NumPy array of binary masks.
    """
    images = []
    annotations = []
    
    # Fetch image files using common extensions.
    image_files = glob.glob(os.path.join(image_folder, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'tiff', 'webp', 'avif'))]
    print(f"Found {len(image_files)} image files in the image folder.")
    
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Look for an annotation file matching: baseName_*_annotation.png
        annotation_files = glob.glob(os.path.join(annotation_folder, f"{base_name}_*_annotation.png"))
        if not annotation_files:
            print(f"Skipping {image_path} due to missing annotation.")
            continue

        annotation_path = annotation_files[0]
        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

        if image is None or annotation is None:
            print(f"Error: Failed to load {image_path} or {annotation_path}.")
            continue

        target_shape = (512, 512)
        image_resized = cv2.resize(image, target_shape)
        annotation_resized = cv2.resize(annotation, target_shape)
        
        # Threshold annotation to ensure binary values (0 or 1)
        _, annotation_thresh = cv2.threshold(annotation_resized, 127, 1, cv2.THRESH_BINARY)
        
        images.append(image_resized)
        annotations.append(annotation_thresh)
    
    images = np.array(images)
    annotations = np.array(annotations)
    
    print(f"Loaded {len(images)} images and {len(annotations)} annotations.")
    if len(images) == 0 or len(annotations) == 0:
        print("Warning: Data not loaded correctly.")
    
    return images, annotations

if __name__ == "__main__":
    X, Y = load_data()
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
