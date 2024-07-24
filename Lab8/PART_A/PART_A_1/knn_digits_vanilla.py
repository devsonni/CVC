import numpy as np
import cv2 as cv

# Load and preprocess the image
image = cv.imread('Imgs/digits.png')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Split the image into 20x20 cells
cell_rows = np.vsplit(gray_image, 50)
cells = [np.hsplit(row, 100) for row in cell_rows]
cells_array = np.array(cells)

# Prepare training and testing data
train_data = cells_array[:, :50].reshape(-1, 400).astype(np.float32)
test_data = cells_array[:, 50:100].reshape(-1, 400).astype(np.float32)

# Generate labels for training and testing data
num_labels = 10
train_labels = np.repeat(np.arange(num_labels), 250)[:, np.newaxis]
test_labels = train_labels.copy()

# Train the KNN model
knn = cv.ml.KNearest_create()
knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)

# Perform KNN classification on the test data
ret, result, neighbours, dist = knn.findNearest(test_data, k=5)

# Calculate and print accuracy
matches = result == test_labels
correct_matches = np.count_nonzero(matches)
accuracy = (correct_matches / result.size) * 100.0
print(f"Accuracy: {accuracy:.2f}%")