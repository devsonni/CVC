import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the image
image = cv.imread('nature.png')

# Reshape the image to a 2D array of pixels
pixel_values = image.reshape((-1, 3)).astype(np.float32)

# Define criteria for k-means clustering
default_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
custom_criteria = (cv.TERM_CRITERIA_MAX_ITER, 50, 1.0)

# Define k values for clustering
professor_k_values = [2, 3, 5, 10, 20, 40]
custom_k_values = [4, 6, 11, 30, 50, 70]

# Set up the subplot grid
rows = 2
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
fig.suptitle('Professor K Values', fontsize=16)

# Apply k-means clustering and plot results for professor k values
for i, k in enumerate(professor_k_values):
    ret, labels, centers = cv.kmeans(pixel_values, k, None, default_criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to uint8 and reshape the segmented image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    
    # Plot the segmented image
    ax = axs[i // cols, i % cols]
    ax.imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB))
    ax.set_title(f'K = {k}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Set up the subplot grid for custom k values
fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
fig.suptitle('Custom K Values', fontsize=16)

# Apply k-means clustering and plot results for custom k values
for i, k in enumerate(custom_k_values):
    ret, labels, centers = cv.kmeans(pixel_values, k, None, custom_criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to uint8 and reshape the segmented image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    
    # Plot the segmented image
    ax = axs[i // cols, i % cols]
    ax.imshow(cv.cvtColor(segmented_image, cv.COLOR_BGR2RGB))
    ax.set_title(f'K = {k}')
    ax.axis('off')

plt.tight_layout()
plt.show()