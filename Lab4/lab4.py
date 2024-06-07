import cv2 as cv
import numpy as np
import shutil
import os
from Wrapper import *

def capture_photo(frame, count, folder_name):
    """Captures and saves the current frame as a photo in the specified folder."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    photo_filename = os.path.join(folder_name, f"photo_{count}.jpg")
    cv.imwrite(photo_filename, frame)
    print(f"Photo captured as {photo_filename}")

def main():
    vid = cv.VideoCapture(0)
    count = 0
    folder_list = os.listdir(".")
    folder_list = [item for item in folder_list if not os.path.isfile(os.path.join(".", item))]
    for folder in folder_list:
        if folder[-1] == "_":
                folder_list.remove(folder)
    folder_list.sort()
    last = folder_list[-1][-1]
    last = int(last) + 1
    folder_name = "Set" + str(last)

    # Set up the window
    window_name = "Photo Booth"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1200, 1080)  # Set the window size to 800x600 pixels
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Show the frame
        cv.imshow(window_name, frame)
        cv.resize(frame, (480//2, 640//2))

        # key reading
        key = cv.waitKey(1)
        
        # Exit on 'Esc' key press
        if key == 27:
            break

        # Capture photo on 'c' key press
        if key == ord('c'):
            capture_photo(frame, count, folder_name)
            count += 1
            for i in range(50):
                key = cv.waitKey(1)
                frame = 255 * np.ones_like(frame, dtype=np.uint8)
                cv.imshow(window_name, frame)
        
        if key == ord('p'):
            vid.release()
            cv.destroyAllWindows()
            imgs, n_imgs = img_reading(folder_name, 1)
            Pipeline(folder_name, imgs, n_imgs)
            print("Panorama Created at --", folder_name, "/Panorama")


    # Release resources
    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
