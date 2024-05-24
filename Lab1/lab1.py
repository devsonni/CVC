import cv2
from datetime import datetime
import numpy as np

def capture_photo(frame):
    """Captures and saves the current frame as a photo."""
    photo_filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(photo_filename, frame)
    print(f"Photo captured as {photo_filename}")

def start_video_recording(vid):
    """Starts video recording and returns the video writer object."""
    video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(vid.get(3)), int(vid.get(4))))
    print("Started recording")
    return out

def stop_video_recording(out):
    """Stops the video recording."""
    out.release()
    print("Stopped recording")

def nothing(x):
    """Dummy function for trackbars."""
    pass

def main():
    # Initialize video capture
    vid = cv2.VideoCapture(0)
    recording = False
    rotate = False
    rotate_angle = 0
    threshold = False
    blur = False
    sharp = False
    extract = False
    out = None

    # Load and resize the logo image
    img = cv2.imread('OpenCV_Logo.png')
    img = cv2.resize(img, (70, 80))

    # Create trackbars for zoom and blur
    cv2.namedWindow('ZoomIn')
    cv2.createTrackbar('Zoom', 'ZoomIn', 0, 90, nothing)
    cv2.namedWindow('Blur')
    cv2.createTrackbar('x', 'Blur', 5, 30, nothing)
    cv2.createTrackbar('y', 'Blur', 5, 30, nothing)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        height, width, channels = frame.shape

        # Read the zoom percentage from the trackbar
        zoom = cv2.getTrackbarPos('Zoom', 'ZoomIn')
        xzoom = int(width * zoom / 200)
        yzoom = int(height * zoom / 200)

        # Crop and resize the frame for zoom effect
        frame = frame[yzoom:height-yzoom, xzoom:width-xzoom]
        frame = cv2.resize(frame, (width, height))

        # Add the current date and time to the frame
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y/%m/%d %H:%M')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, formatted_datetime, (300, 450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Make ROI and swap pixels at the top part of the frame
        roi = frame[400:480, 280:640]
        frame[0:80, 280:280+360, :] = roi

        # Add rectangle around the frame
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

        # Blend OpenCV logo with camera feed
        blended = cv2.addWeighted(frame[0:80, 0:70], 0.6, img, 0.4, 0)
        frame[0:80, 0:70] = blended

        # Write the frame if recording
        if recording and out:
            out.write(frame)

        # Check for keypress
        key = cv2.waitKey(1)

        # Rotate frame on 'r' key press
        if key == ord('r'):
            rotate = True
            rotate_angle += 10

        if rotate:
            M = cv2.getRotationMatrix2D((240, 320), rotate_angle, 1)
            frame = cv2.warpAffine(frame, M, (width, height))

        # Toggle thresholding on 't' key press
        if key == ord('t'):
            threshold = not threshold

        if threshold:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)

        # Toggle blurring on 'b' key press
        if key == ord('b'):
            blur = not blur

        if blur:
            x = cv2.getTrackbarPos('x', 'Blur')
            y = cv2.getTrackbarPos('y', 'Blur')
            frame = cv2.GaussianBlur(frame, (x, y), 0)

        # Toggle sharpening on 's' key press
        if key == ord('s'):
            sharp = not sharp

        if sharp:
            frame = cv2.bilateralFilter(frame, 12, 100, 100)

        # Toggle color extraction on 'e' key press
        if key == ord('e'):
            extract = not extract

        if extract:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Show the frame
        cv2.imshow("Photo Booth", frame)

        # Exit on 'Esc' key press
        if key == 27:
            break
        # Capture photo on 'c' key press
        elif key == ord('c'):
            capture_photo(frame)
            for i in range(50):
                key = cv2.waitKey(1)
                frame = 255 * np.ones_like(frame, dtype=np.uint8)
                cv2.imshow("Photo Booth", frame)
        # Start/stop video recording on 'v' key press
        elif key == ord('v'):
            if recording:
                stop_video_recording(out)
                recording = False
            else:
                out = start_video_recording(vid)
                recording = True

    # Release resources
    vid.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()