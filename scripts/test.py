from skimage.metrics import structural_similarity as ssim
import cv2

# Load your images
image1 = cv2.imread('./gt/0008.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./gt/0012.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure both images have the same dimensions
height, width = image1.shape
image2 = cv2.resize(image2, (width, height))

# Calculate SSIM
ssim_value, _ = ssim(image1, image2, full=True)

print(f'SSIM Value: {ssim_value}')
