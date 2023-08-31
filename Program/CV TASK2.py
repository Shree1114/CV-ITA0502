import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = ""
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)


b, g, r = cv2.split(image)

# Calculate histograms for each color channel
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the input image
axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Input Image')
axs[0, 0].axis('off')

# Plot the histograms
axs[0, 1].plot(hist_b, color='blue')
axs[0, 1].set_title('Blue Histogram')
axs[0, 1].set_xlabel('Bins')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].plot(hist_g, color='green')
axs[1, 0].set_title('Green Histogram')
axs[1, 0].set_xlabel('Bins')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].plot(hist_r, color='red')
axs[1, 1].set_title('Red Histogram')
axs[1, 1].set_xlabel('Bins')
axs[1, 1].set_ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
