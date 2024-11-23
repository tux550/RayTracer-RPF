# UTIL for sanity check: Compare images

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
    difference_count = 0
    for i in range(len(img1_array)):
        for j in range(len(img1_array[0])):
            if not np.array_equal(img1_array[i][j], img2_array[i][j]):
                #print(f"Pixels at ({i},{j}) are different.")
                #print(f"Pixel 1: {img1_array[i][j]}")
                #print(f"Pixel 2: {img2_array[i][j]}")
                
                difference_count += 1
    print(f"Number of different pixels: {difference_count} out of {len(img1_array)*len(img1_array[0])}")
    return difference_count == 0
    #return np.array_equal(img1_array, img2_array)

# Example usage
file1 = 'output/sample.png'
file2 = 'output/sample_unfiltered.png'

print("Starting comparison...")
try:
  if are_images_identical(file1, file2):
      print("The PNG files are different.")
  else:
      print("The PNG files are the same.")
except Exception as e:
  print("Error: ", e)
print("Comparison finished.")