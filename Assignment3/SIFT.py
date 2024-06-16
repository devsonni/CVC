import cv2
import numpy as np
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt

# Image dimensions and constants
IMAGE_ROWS = 256
IMAGE_COLS = 256
OCTAVES = 3
LEVELS = 3
SIGMA_INITIAL = np.sqrt(2)

# Read and preprocess the image
image = cv2.imread('lenna.png')
original_image = image.copy()
print(f"Original image shape: {image.shape}")

# Resize, convert to grayscale and normalize the image
image = cv2.resize(image, (IMAGE_ROWS, IMAGE_COLS))
print(f"Resized image shape: {image.shape}")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_normalized = cv2.normalize(image_gray.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Scale Space extrema Detection
# Creating a Difference of Gaussian (DoG) layered matrix
dog_layers = [np.zeros((int(IMAGE_ROWS * np.float_power(2, 2-i)) + 2, int(IMAGE_COLS * np.float_power(2, 2-i)) + 2, LEVELS)) for i in range(1, OCTAVES + 1)]

# Initial image upsampled and padded
temp_img = cv2.resize(image_normalized, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
temp_img = cv2.copyMakeBorder(temp_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

## Iterating through all octaves and levels to create a DoG stack of images
for i in range(1, OCTAVES + 1):
    dog_layer = dog_layers[i-1]
    fig, axes = plt.subplots(1, LEVELS, figsize=(15, 3))
    
    for j in range(1, LEVELS + 1):
        scale = SIGMA_INITIAL * np.float_power(np.sqrt(2), (1 / LEVELS)) ** ((i-1) * LEVELS + j)
        kernel_size = int(np.floor(6 * scale))
        gaussian_kernel = cv2.getGaussianKernel(kernel_size, scale)
        
        img_blur = convolve(temp_img, gaussian_kernel.reshape(1, kernel_size), mode='same')
        img_blur = convolve(img_blur, gaussian_kernel.reshape(1, kernel_size), mode='same')
        dog_layer[:, :, j-1] = img_blur - temp_img
        temp_img = img_blur

        if j == LEVELS:
            temp_img = temp_img[1:-2, 1:-2]

        axes[j-1].imshow(dog_layer[:, :, j-1], cmap='gray')
        axes[j-1].set_title(f'Octave {i}, Level {j}')

    fig.suptitle(f'Difference of Gaussian (DoG) Stack - Octave {i}', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'dog_stack_octave_{i}.png')  # Save the figure
    plt.show()  # Display the figure

    dog_layers[i-1] = dog_layer
    temp_img = temp_img[::2, ::2]
    temp_img = cv2.copyMakeBorder(temp_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

# Continue with keypoint localization and further processing
# Keypoint Localization
interval = LEVELS - 1
extrema = []

for i in range(1, OCTAVES + 1):
    m, n, _ = dog_layers[i-1].shape
    m -= 2
    n -= 2
    search_space = int(m * n / np.float_power(4, (i-1)))
    
    for k in range(2, interval + 1):
        for j in range(1, search_space + 1):
            x = int((j-1) / n) + 1
            y = int((j-1) % n) + 1
            
            sub_img = dog_layers[i-1][x:x+3, y:y+3, k-2:k+1]
            max_val = np.max(sub_img)
            min_val = np.min(sub_img)
            
            if max_val == dog_layers[i-1][x, y, k-1]:
                extrema.append([i, k-1, j, 1])
            if min_val == dog_layers[i-1][x, y, k-1]:
                extrema.append([i, k-1, j, -1])

# Extracting extrema and rejecting all others
extrema = np.array(extrema).flatten()
extrema = extrema[extrema != 0]

# Reconstruction of x and y from the given volume location and octave
img_height, img_width = image_gray.shape
extrema_values = extrema[2::4]
extrema_octaves = extrema[0::4]

x = np.floor((extrema_values - 1) / (img_width / np.float_power(2, extrema_octaves-2))) + 1
y = np.remainder((extrema_values - 1), (img_height / np.float_power(2, extrema_octaves-2))) + 1

ry = y / np.float_power(2, OCTAVES - 1 - extrema_octaves)
rx = x / np.float_power(2, OCTAVES - 1 - extrema_octaves)

# Plotting the extremas found
plt.figure()
plt.imshow(image_gray, cmap='gray')
plt.scatter(ry, rx, marker='+', color='blue')
plt.title('Initial Keypoints')  # Save the figure
plt.show()

# Accurate Keypoint Localization
threshold = 0.1
r = 10
extrema_volume = len(extrema) // 4

m, n = image_gray.shape
second_order_x = convolve([[-1, 1], [-1, 1]], [[-1, 1], [-1, 1]])
second_order_y = convolve([[-1, -1], [1, 1]], [[-1, -1], [1, 1]])

for i in range(1, OCTAVES + 1):
    for j in range(1, LEVELS + 1):
        dog_img = dog_layers[i-1][:, :, j-1]
        temp = -1 / convolve(dog_img, second_order_y, mode='same') * convolve(dog_img, [[-1, -1], [1, 1]], mode='same')
        dog_layers[i-1][:, :, j-1] = temp * convolve(dog_img, [[-1, -1], [1, 1]], mode='same') * 0.5 + dog_img

count = 0
for i in range(1, int(extrema_volume + 1)):
    x = int(np.floor((extrema[4*(i-1)+2] - 1) / (n / np.float_power(2, extrema[4*(i-1)] - 2))) + 1)
    y = int(np.remainder((extrema[4*(i-1)+2] - 1), (m / np.float_power(2, extrema[4*(i-1)] - 2))) + 1)
    rx = int(x + 1)
    ry = int(y + 1)
    rz = int(extrema[4*(i-1)+1])
    
    z = dog_layers[int(extrema[4*(i-1)])-1][rx-1, ry-1, rz]
    if np.abs(z) < threshold:
        extrema[4*(i-1)+3] = 0
        count += 1

print(f"Number of extrema below threshold: {count}")

idx = np.where(extrema == 0)[0]
idx = np.concatenate([idx, idx-1, idx-2, idx-3])
extrema = np.delete(extrema, idx)

# Extracting and plotting the better extrema
extrema_volume = len(extrema) // 4
extrema_values = extrema[2::4]
extrema_octaves = extrema[0::4]

x = np.floor((extrema_values - 1) / (img_width / np.float_power(2, extrema_octaves-2))) + 1
y = np.remainder((extrema_values - 1), (img_height / np.float_power(2, extrema_octaves-2))) + 1

ry = y / np.float_power(2, OCTAVES - 1 - extrema_octaves)
rx = x / np.float_power(2, OCTAVES - 1 - extrema_octaves)

plt.figure()
plt.imshow(image_gray, cmap='gray')
plt.scatter(ry, rx, marker='+', color='green')
plt.title('Filtered Keypoints')
plt.show()

count2 = 0
for i in range(1, int(extrema_volume + 1)):
    x = int(np.floor((extrema[4*(i-1)+2] - 1) / (n / np.float_power(2, extrema[4*(i-1)] - 2))) + 1)
    y = int(np.remainder((extrema[4*(i-1)+2] - 1), (m / np.float_power(2, extrema[4*(i-1)] - 2))) + 1)
    rx = int(x + 1)
    ry = int(y + 1)
    rz = int(extrema[4*(i-1)+1])
    
    # Calculating the double derivatives using David Lowe's method
    Dxx = dog_layers[int(extrema[4*(i-1)])-1][rx-2, ry-1, rz] + dog_layers[int(extrema[4*(i-1)])-1][rx, ry-1, rz] - 2 * dog_layers[int(extrema[4*(i-1)])-1][rx-1, ry-1, rz]
    Dyy = dog_layers[int(extrema[4*(i-1)])-1][rx-1, ry-2, rz] + dog_layers[int(extrema[4*(i-1)])-1][rx-1, ry, rz] - 2 * dog_layers[int(extrema[4*(i-1)])-1][rx-1, ry-1, rz]
    Dxy = dog_layers[int(extrema[4*(i-1)])-1][rx-2, ry-2, rz] + dog_layers[int(extrema[4*(i-1)])-1][rx, ry, rz] - dog_layers[int(extrema[4*(i-1)])-1][rx-2, ry, rz] - dog_layers[int(extrema[4*(i-1)])-1][rx, ry-2, rz]
    
    determinant = Dxx * Dyy - Dxy * Dxy
    trace = Dxx + Dyy
    R = trace**2 / determinant
    R_threshold = (r + 1)**2 / r
    
    if determinant < 0 or R > R_threshold:
        extrema[4*(i-1)+3] = 0
        count2 += 1

print(f"Number of extrema removed after second thresholding: {count2}")

idx = np.where(extrema == 0)[0]
idx = np.concatenate([idx, idx-1, idx-2, idx-3])
extrema = np.delete(extrema, idx)

# Extracting and plotting the final extrema
extrema_volume = len(extrema) // 4
extrema_values = extrema[2::4]
extrema_octaves = extrema[0::4]

x = np.floor((extrema_values - 1) / (img_width / np.float_power(2, extrema_octaves-2))) + 1
y = np.remainder((extrema_values - 1), (img_height / np.float_power(2, extrema_octaves-2))) + 1

ry = y / np.float_power(2, OCTAVES - 1 - extrema_octaves)
rx = x / np.float_power(2, OCTAVES - 1 - extrema_octaves)

plt.figure()
plt.imshow(image_gray, cmap='gray')
plt.scatter(ry, rx, marker='+', color='red')
plt.title('Final Keypoints')
plt.show()
