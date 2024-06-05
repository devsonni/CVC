import cv2
from datetime import datetime
import numpy as np
import time
from scipy.ndimage import convolve

def capture_photo(frame):
    """Captures and saves the current frame as a photo."""
    photo_filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(photo_filename, frame)
    print(f"Photo captured as {photo_filename}")

def start_video_recording(vid):
    """Starts video recording and returns the video writer object."""
    video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(640), int(480)))
    print("Started recording")
    return out

def stop_video_recording(out):
    """Stops the video recording."""
    out.release()
    print("Stopped recording")

def nothing(x):
    """Dummy function for trackbars."""
    pass

def custom_sobel_fun(frame):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # Apply the Sobel filters using filter2D
    gradient_x = cv2.filter2D(gray_image, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(gray_image, cv2.CV_64F, sobel_y)
    
    # Normalize the gradient images to the range [0, 255]
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)
    
    gradient_x = (gradient_x / np.max(gradient_x)) * 255
    gradient_y = (gradient_y / np.max(gradient_y)) * 255
    
    gradient_x = gradient_x.astype(np.uint8)
    gradient_y = gradient_y.astype(np.uint8)
    
    return gradient_x, gradient_y

def custom_laplacian(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    
    # Apply the Laplacian filter using filter2D
    laplacian_image = cv2.filter2D(gray_image, cv2.CV_64F, laplacian_kernel)
    
    # Normalize the Laplacian image to the range [0, 255]
    laplacian_image = np.abs(laplacian_image)
    laplacian_image = (laplacian_image / np.max(laplacian_image)) * 255
    laplacian_image = laplacian_image.astype(np.uint8)
    
    return laplacian_image

def main():
    time.sleep(2)

    # Initialize video capture
    vid = cv2.VideoCapture(0)
    recording = False
    sobelx = False
    sobely = False
    canny = False
    custom_sobel = False
    out = None
    four_pressed = False

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        if not canny:
            if sobelx and not sobely:
                x = cv2.getTrackbarPos('sobel_x', 'Sobel X')
                frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=int(2*x+1))
                sobely = False
        
        if not canny:
            if sobely and not sobelx:
                y = cv2.getTrackbarPos('sobel_y', 'Sobel Y')
                frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=int(2*y+1))
                sobelx = False
        
        if canny:
            thres1 = cv2.getTrackbarPos('Threshold 1', 'Threshold')
            thres2 = cv2.getTrackbarPos('Threshold 2', 'Threshold')
            frame = cv2.Canny(frame, thres1, thres2, apertureSize=5)
        
        if four_pressed:
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (width//2, height//2)) 
            cv2.imshow("Original", frame)
            lap = custom_laplacian(frame.copy())
            sobx, soby = custom_sobel_fun(frame.copy())
            cv2.imshow("Laplacian", lap)
            cv2.imshow("Sobel X", sobx)
            cv2.imshow("Sobel Y", soby)
            # Move windows to specified locations
            cv2.moveWindow('Original', 0, 0)  # Top-left corner
            cv2.moveWindow('Laplacian', 0, frame.shape[0]+100)  # Below the original frame
            cv2.moveWindow('Sobel X', frame.shape[1]+50, 0)  # Next to the first camera
            cv2.moveWindow('Sobel Y', frame.shape[1]+50, frame.shape[0]+100)  # Below the original frame of the second camera


        # Check for keypress
        key = cv2.waitKey(1)
        if key != -1:
            # Print the character of the key pressed
            print("PRESSED KEY:", chr(key))

        if key == ord('s'):

            next_key = cv2.waitKey(0)  # Wait for the next key press            
            print("Next key:", next_key)  # Debugging output

            if next_key == ord('x'):
                cv2.namedWindow('Sobel X')
                cv2.createTrackbar('sobel_x', 'Sobel X', 0, 15, nothing)
                sobely = False
                sobelx = True

            elif next_key == ord('y'):
                cv2.namedWindow('Sobel Y')
                cv2.createTrackbar('sobel_y', 'Sobel Y', 0, 15, nothing)
                sobelx = False
                sobely = True

        if key == ord('d'):
            canny = True
            cv2.namedWindow('Threshold')
            cv2.createTrackbar('Threshold 1', 'Threshold', 1, 5000, nothing)
            cv2.createTrackbar('Threshold 2', 'Threshold', 1, 5000, nothing)

        if key == ord('4'):
            four_pressed = True


        # Show the frame
        cv2.imshow("Photo Booth", frame)

        # Write the frame if recording
        if recording and out:
            out.write(frame)

        # Exit on 'Esc' key press
        if key == 27:
            break
        # Capture photo on 'c' key press
        if key == ord('c'):
            capture_photo(frame)
            for i in range(50):
                key = cv2.waitKey(1)
                frame = 255 * np.ones_like(frame, dtype=np.uint8)
                cv2.imshow("Photo Booth", frame)
        # Start/stop video recording on 'v' key press
        if key == ord('v'):
            if recording:
                stop_video_recording(out)
                recording = False
            else:
                out = start_video_recording(frame)
                recording = True

    # Release resources
    vid.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()