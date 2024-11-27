# UTIL for sanity check: Compare images

from re import A
from PIL import Image
import numpy as np

def are_images_identical(image_path1, image_path2):
    # Open both images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    # Convert images to RGB to ensure compatibility (if they have alpha channel)
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    
    # Check if the size is the same
    if img1.size != img2.size:
        return False
    
    # Convert images to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Compare the arrays
    different_pixels = 0
    for i in range(img1_array.shape[0]):
        for j in range(img1_array.shape[1]):
            for k in range(img1_array.shape[2]):
                if img1_array[i, j, k] != img2_array[i, j, k]:
                    different_pixels += 1
    print(f"Number of different pixels: {different_pixels} / {img1_array.shape[0] * img1_array.shape[1]}")
    print(f"(%: {100 * different_pixels / (img1_array.shape[0] * img1_array.shape[1])})")

# Example usage
file1 = 'output/sample.png'
file2 = 'output/sample_unfiltered.png'


are_images_identical(file1, file2)