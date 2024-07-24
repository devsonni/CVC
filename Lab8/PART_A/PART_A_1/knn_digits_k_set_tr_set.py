import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

# Load and preprocess the image
image = cv.imread('Imgs/digits.png')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Split the image into 20x20 cells
cell_rows = np.vsplit(gray_image, 50)
cells = [np.hsplit(row, 100) for row in cell_rows]
cells_array = np.array(cells)

# Initialize variables for storing results
train_size_percentages = np.arange(10, 100, 10)
results = {'split': [], 'k': [], 'accuracy': []}

# Prepare and evaluate KNN classifier for different train/test splits and k values
for train_size in train_size_percentages:
    train_data = cells_array[:, :train_size].reshape(-1, 400).astype(np.float32)
    test_data = cells_array[:, train_size:100].reshape(-1, 400).astype(np.float32)

    num_labels = 10
    train_labels = np.repeat(np.arange(num_labels), train_data.shape[0] / num_labels)[:, np.newaxis]
    test_labels = np.repeat(np.arange(num_labels), test_data.shape[0] / num_labels)[:, np.newaxis]

    k_values = np.arange(1, 10)
    for k in k_values:
        knn = cv.ml.KNearest_create()
        knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = knn.findNearest(test_data, k=k)
        
        matches = result == test_labels
        correct_matches = np.count_nonzero(matches)
        accuracy = (correct_matches / result.size) * 100.0
        
        results['split'].append(train_size)
        results['k'].append(k)
        results['accuracy'].append(accuracy)
        
        print(f"Train size: {train_size}%, k={k}, Accuracy: {accuracy:.2f}%")

# Convert results to a DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Plot accuracy for different train/test splits and k values
for train_size in train_size_percentages:
    data_subset = results_df[results_df['split'] == train_size]
    plt.plot(data_subset['k'].values, data_subset['accuracy'].values, label=f'Train size: {train_size}%')

plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.title('KNN Accuracy for Different Train/Test Splits and k Values')
plt.legend()
plt.show()
