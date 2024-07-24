import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data and convert the letters to numbers
data = np.loadtxt('letter_recognition/letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch) - ord('A')})

# Initialize variables for storing results
train_size_percentages = np.arange(0.1, 1, 0.1)
results = {'split': [], 'k': [], 'accuracy': []}

# Prepare and evaluate KNN classifier for different train/test splits and k values
for train_size in train_size_percentages:
    # Split the dataset as per the percentage
    train_data = data[0:int(train_size * data.shape[0]), :]
    test_data = data[int(train_size * data.shape[0]):, :]

    # Split train_data and test_data into features and responses
    responses, train_features = np.hsplit(train_data, [1])
    labels, test_features = np.hsplit(test_data, [1])

    k_values = np.arange(1, 10)
    for k in k_values:
        knn = cv.ml.KNearest_create()
        knn.train(train_features, cv.ml.ROW_SAMPLE, responses)
        ret, result, neighbours, dist = knn.findNearest(test_features, k=k)
        
        correct_matches = np.count_nonzero(result == labels)
        accuracy = (correct_matches / result.size) * 100.0
        
        results['split'].append(train_size)
        results['k'].append(k)
        results['accuracy'].append(accuracy)
        
        print(f"Train size: {train_size:.1f}, k={k}, Accuracy: {accuracy:.2f}%")

# Convert results to a DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Plot accuracy for different train/test splits and k values
for train_size in train_size_percentages:
    data_subset = results_df[results_df['split'] == train_size]
    plt.plot(data_subset['k'].values, data_subset['accuracy'].values, label=f'Train size: {int(train_size*100)}%')

plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.title('KNN Accuracy for Different Train/Test Splits and k Values')
plt.legend()
plt.show()
