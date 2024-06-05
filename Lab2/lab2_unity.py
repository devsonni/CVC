import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

# Reading imgs form the given directory, this will be replaced in server implementation
def img_reading(folder):
    
    image_list = os.listdir(folder)
    image_list = [item for item in image_list if os.path.isfile(os.path.join(folder, item))]
    
    return image_list

def display_image(image, title):
  
    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Determine the new dimensions of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust the rotation matrix to take into account the translation due to padding
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    
    # Apply rotation to the image with padding
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated_image

def calculate_bounding_box(pts1, pts2):
    # Combine the original and desired points
    all_points = np.concatenate((pts1, pts2), axis=0)
    
    # Find the minimum and maximum x and y coordinates
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])
    
    # Calculate the new width and height
    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)
    
    return new_width, new_height

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    # Read the image using OpenCV
    image = cv2.imread("UnityHall.jpg")

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Titles for each image
    titles = ['Original', 'Scaled Up', 'Affine', 'Rotated', 'Scaled Down', 'Perspective']

    ## scalling up
    height, width, _ = image_rgb.shape
    scaled_up = cv.resize(image_rgb, (int(width+(20*width/100)), int((20*width/100)+height)), interpolation = cv.INTER_CUBIC)
    
    ## affine
    pts1 = np.float32([[0, 0], [height, 0], [height, width]])
    pts2 = np.float32([[0,50],[height,0],[height,width-50]])
    newwidth, newheight = calculate_bounding_box(pts1, pts2)

    M = cv.getAffineTransform(pts1,pts2)
    
    Affine = cv.warpAffine(image_rgb,M,(newwidth-20, newheight-150), cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

    ## Rotated
    # M = cv.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0), 10, 1)
    # rotated = cv.warpAffine(image_rgb, M,(width,height))
    rotated = rotate_image(image_rgb, 10)

    ## Scale Down
    # scalling up
    height, width, _ = image_rgb.shape
    scaled_down = cv.resize(image_rgb, (int(width-(20*width/100)), int(height-(20*width/100))), interpolation = cv.INTER_CUBIC)

    ## Perspective transform
    pts1_ = np.float32([[0,0],[0,height],[width,0],[width, height]])
    pts2_ = np.float32([[0,0],[0,height-60],[width+20,0],[width, height]])
    
    M = cv.getPerspectiveTransform(pts1_, pts2_)
    
    newwidth, newheight = calculate_bounding_box(pts1_, pts2_)
    perspective = cv.warpPerspective(image_rgb,M,(newwidth, newheight), cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Display each image separately
    display_image(image_rgb, titles[0])
    display_image(scaled_up, titles[1])
    display_image(Affine, titles[2])
    display_image(rotated, titles[3])
    display_image(scaled_down, titles[4])
    display_image(perspective, titles[5])

    corner_detection_folder = "corner_detection"

    # Run Harris corner detector on each image and save the results
    list_img = img_reading("Images")

    for file in list_img:
        img = cv.imread("Images/"+file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [255, 0, 0]  # Marking corners in red
        save_image(img, corner_detection_folder, f"{file}.jpg")

    for file in list_img:
        img = cv2.imread(os.path.join("Images", file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Create a SIFT object
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Draw keypoints on the image
        img_with_keypoints = img.copy()
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.MARKER_CROSS)
        
        # Draw plus signs at keypoints
        for kp in keypoints:
            x, y = kp.pt  
            x = int(round(x))  # Round an cast to int
            y = int(round(y))

            # Draw a cross with (x, y) center
            img_with_keypoints = cv2.drawMarker(img_with_keypoints, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

        # Save the image with keypoints
        save_image(img_with_keypoints, "sift_detection", f"{file}.jpg")

if __name__ == "__main__":
    main()