import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

# Evaluate KNN classifier for different values of k
k_values = np.arange(1, 10)
accuracies = []

for k in k_values:
    knn = cv.ml.KNearest_create()
    knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test_data, k=k)
    
    matches = result == test_labels
    correct_matches = np.count_nonzero(matches)
    accuracy = (correct_matches / result.size) * 100.0
    accuracies.append(accuracy)
    
    print(f"k={k}, Accuracy: {accuracy:.2f}%")

# Plotting the accuracy for different values of k
plt.plot(k_values, accuracies)
plt.xlabel('k-values')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. k-values for KNN Classifier')
plt.show()
