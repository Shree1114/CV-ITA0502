import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Load an image from file
path = ""
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
canny = cv2.Canny(image, 100, 200)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Define Prewitt edge detection kernels
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Apply Prewitt edge detection
prewitt_x_result = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
prewitt_y_result = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
prewitt_magnitude = np.sqrt(prewitt_x_result**2 + prewitt_y_result**2)
prewitt_normalized = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Define Robert's edge detection kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

# Apply Robert's edge detection
roberts_x_result = cv2.filter2D(image, cv2.CV_64F, roberts_x)
roberts_y_result = cv2.filter2D(image, cv2.CV_64F, roberts_y)
roberts_magnitude = np.sqrt(roberts_x_result**2 + roberts_y_result**2)
roberts_normalized = cv2.normalize(roberts_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Calculate Mean Squared Error (MSE)
mse_sobel = np.mean((image - sobel_magnitude)**2)
mse_canny = np.mean((image - canny)**2)
mse_prewitt = np.mean((image - prewitt_magnitude)**2)
mse_robert = np.mean((image - roberts_magnitude)**2)

# Calculate Structural Similarity Index (SSI)
ssi_sobel = ssim(image, sobel_magnitude)
ssi_canny = ssim(image, canny)
ssi_prewitt = ssim(image, prewitt_magnitude)
ssi_robert = ssim(image, roberts_magnitude)

print(f"MSE (Sobel): {mse_sobel:.2f}, MSE (Canny): {mse_canny:.2f},MSE (Robert): {mse_robert:.2f}, MSE (prewit): {mse_prewitt:.2f}")
print(f"SSI (Sobel): {ssi_sobel:.2f}, SSI (Canny): {ssi_canny:.2f},SSI (Robert): {ssi_robert:.2f}, SSI (Prewitt): {ssi_prewitt:.2f}")

# Display the original and edge-detected images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(2, 3, 2), plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Detection'), plt.axis('off')

plt.subplot(2, 3, 3), plt.imshow(sobel_normalized, cmap='gray')
plt.title('Sobel Edge Detection'), plt.axis('off')

plt.subplot(2, 3, 4), plt.imshow(prewitt_normalized, cmap='gray')
plt.title('Prewitt Edge Detection'), plt.axis('off')

plt.subplot(2, 3, 5), plt.imshow(roberts_normalized, cmap='gray')
plt.title("Robert's Edge Detection"), plt.axis('off')

plt.tight_layout()
plt.show()
