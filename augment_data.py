import os
import random
from PIL import Image
import numpy as np
from dataset_loader import load_data  # Optional: for testing purposes
import cv2


def augment_image(img):
    """
    Applies a random transformation to an image.
    
    Parameters:
        img (PIL.Image): The image to augment.
        
    Returns:
        PIL.Image: The augmented image.
    """
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transformations = [flip_image, rotate_image, crop_image]
    transformation = random.choice(transformations)
    return transformation(img)

def augment_annotation(mask):
    """
    Applies a random geometric transformation to an annotation mask.
    For segmentation masks, only geometric transforms (flip, rotate, crop) should be applied.
    
    Parameters:
        mask (PIL.Image): The mask to augment.
        
    Returns:
        PIL.Image: The augmented mask.
    """
    # Convert mask to mode 'L' (grayscale) if not already
    if mask.mode != 'L':
        mask = mask.convert('L')
    # For consistency, use the same transformation options as for images.
    transformations = [flip_image, rotate_image, crop_image]
    transformation = random.choice(transformations)
    return transformation(mask)

def flip_image(img):
    """Flips the image horizontally."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image(img):
    """Rotates the image by a random angle between -30 and 30 degrees."""
    angle = random.randint(-30, 30)
    return img.rotate(angle, resample=Image.BILINEAR)

def crop_image(img):
    """Crops the image to a random area and resizes back to 512x512."""
    width, height = img.size
    # Choose a random crop box (ensuring minimum size)
    left = random.randint(0, width // 4)
    upper = random.randint(0, height // 4)
    right = random.randint(3 * width // 4, width)
    lower = random.randint(3 * height // 4, height)
    cropped = img.crop((left, upper, right, lower))
    # Resize cropped image back to 512x512
    return cropped.resize((512, 512), Image.BILINEAR)

def augment_dataset(X, Y, output_dir="augmented_data"):
    """
    Augments the dataset by applying random transformations to images and masks.
    
    Parameters:
        X (np.array): Array of images (as NumPy arrays).
        Y (np.array): Array of masks (as NumPy arrays).
        output_dir (str): Directory where augmented images will be saved.
        
    Returns:
        Tuple (X_aug, Y_aug) as NumPy arrays.
    """
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be a string")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    X_aug = []
    Y_aug = []
    
    # Convert NumPy arrays to PIL Images
    for i in range(len(X)):
        img = Image.fromarray(X[i])
        mask = Image.fromarray((Y[i]*255).astype(np.uint8))  # Convert binary mask back to 0-255 for augmentation
        
        # Generate 5 augmented versions per image
        for j in range(5):
            aug_img = augment_image(img)
            aug_mask = augment_annotation(mask)
            # Convert back to NumPy array; for mask, re-threshold to binary
            aug_img_np = np.array(aug_img)
            aug_mask_np = np.array(aug_mask)
            _, aug_mask_np = cv2.threshold(aug_mask_np, 127, 1, cv2.THRESH_BINARY)
            
            # Save augmented images (optional)
            aug_img_filename = f"{os.path.splitext(f'image_{i}')[0]}_aug_{j}.jpg"
            aug_mask_filename = f"{os.path.splitext(f'mask_{i}')[0]}_aug_{j}.png"
            aug_img_path = os.path.join(output_dir, aug_img_filename)
            aug_mask_path = os.path.join(output_dir, aug_mask_filename)
            Image.fromarray(aug_img_np).save(aug_img_path)
            Image.fromarray((aug_mask_np*255).astype(np.uint8)).save(aug_mask_path)
            
            X_aug.append(aug_img_np)
            Y_aug.append(aug_mask_np)
    
    X_aug = np.array(X_aug)
    Y_aug = np.array(Y_aug)
    
    print(f"Generated {len(X_aug)} augmented images and {len(Y_aug)} augmented masks.")
    return X_aug, Y_aug

# For testing, you can run this script directly.
if __name__ == "__main__":
    X, Y = load_data()
    X_aug, Y_aug = augment_dataset(X, Y, output_dir="augmented_data")
