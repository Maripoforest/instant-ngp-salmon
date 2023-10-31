import cv2
import numpy as np

# Load the two images (replace 'image1.jpg' and 'image2.jpg' with your image file paths)
motherpath = "/home/socialai/instant-ngp-salmon/snapshots_delay/video0010"

for i in range(50):
    
    image1 = cv2.imread('image1.jpg')
    image2 = cv2.imread('image2.jpg')

# Ensure that the images have the same dimensions
if image1.shape != image2.shape:
    raise ValueError("The images must have the same dimensions.")

# Calculate the Mean Squared Error (MSE)
mse = np.mean((image1 - image2) ** 2)

# Calculate the maximum possible pixel value
max_pixel_value = 255  # For 8-bit images

# Calculate the PSNR
psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

print(f"PSNR: {psnr} dB")