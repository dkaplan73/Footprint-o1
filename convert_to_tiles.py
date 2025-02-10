import cv2
import numpy as np
import os

# Define tile size (in pixels)
TILE_SIZE = 16

def convert_mask_to_tiles(mask_path, output_dir="tiles/"):
    """
    Converts a segmentation mask into a tile-based format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask from {mask_path}")
        return
        
    h, w = mask.shape

    for i in range(0, h, TILE_SIZE):
        for j in range(0, w, TILE_SIZE):
            tile = mask[i:i+TILE_SIZE, j:j+TILE_SIZE]
            tile_filename = os.path.join(output_dir, f"tile_{i//TILE_SIZE}_{j//TILE_SIZE}.png")
            cv2.imwrite(tile_filename, tile)

    print(f"Tiles saved in {output_dir}")

# Example usage:
if __name__ == "__main__":
    # Replace with the actual mask file you wish to tile
    convert_mask_to_tiles("output_masks/mask_sample.png")
