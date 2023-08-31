import cv2
import numpy as np

# Load the image
image = cv2.imread("")
img = cv2.imread("")

# Create a mask to indicate the initial state (0 for background, 1 for foreground)
mask = np.zeros(img.shape[:2], np.uint8)

# Define the region of interest (ROI) where the foreground object is located
rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100)

# Apply the GrabCut algorithm
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to separate the background (0 and 2) from the foreground (3 and 1)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Create a new image by multiplying the mask with the original image
result = img * mask2[:, :, np.newaxis]

# Display the result
cv2.imshow('Foreground Removed', result)
cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
