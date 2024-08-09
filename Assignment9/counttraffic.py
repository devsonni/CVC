import sys
import cv2
import numpy as np
from collections import defaultdict  # Import defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python yolo_tracking.py <video_path>")
    sys.exit(1)
video_path = sys.argv[1]  # Get video path from command line argument

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = video_path.split('.')[0] + '_tracked.mp4'
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

# Initialize Matplotlib figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Initialize object counts and tracked object IDs
object_counts = defaultdict(int)
tracked_ids = set()

# Initialize a dictionary to keep track of object IDs and their last seen frame index
last_seen_frame = {}
max_disappeared_frames = 30  # Maximum number of frames an object can disappear and still be considered the same

frame_idx = 0

# Define the Region of Interest (ROI)
roi_x1, roi_y1 = 1400-15, 0  # Top-left corner of the ROI
roi_x2, roi_y2 = 1500, 900  # Bottom-right corner of the ROI

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)

        # Remove IDs of objects that have been gone for too long
        for obj_id in list(last_seen_frame):
            if frame_idx - last_seen_frame[obj_id] > max_disappeared_frames:
                tracked_ids.discard(obj_id)
                del last_seen_frame[obj_id]

        # Draw the Crosswalk detection
        cv2.rectangle(frame, (600, 600), (1720, 1000), (0, 0, 0), 5)

        # this is for counting
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Bounding box coordinates
                conf = bbox.conf[0]  # Confidence score
                cls = int(bbox.cls[0])  # Class ID
                obj_id = int(bbox.id[0])  # Object ID

                # Check if the object's bounding box is within the ROI
                if x1 >= roi_x1 and y1 >= roi_y1 and x1 <= roi_x2 and y1 <= roi_y2:
                    # Only update object count if the object ID hasn't been seen before or was seen too long ago
                    if obj_id not in tracked_ids:
                        tracked_ids.add(obj_id)
                        object_counts[model.names[cls]] += 1

                    # Update the last seen frame index for the current object ID
                    last_seen_frame[obj_id] = frame_idx

                    # Draw the bounding box within the ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw the label
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # this is for drawing
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Bounding box coordinates
                conf = bbox.conf[0]  # Confidence score
                cls = int(bbox.cls[0])  # Class ID
                obj_id = int(bbox.id[0])  # Object ID

                # Check if the object's bounding box is within the ROI
                if x1 >= 600 and y1 >= 0 and x1 <= 1720 and y1 <= 1000:
                    # Only update object count if the object ID hasn't been seen before or was seen too long ago

                    # Draw the bounding box within the ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the cumulative counts on the frame
        y_offset = 30
        for obj, count in object_counts.items():
            count_label = f"{obj}: {count}"
            cv2.putText(frame, count_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 30

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame using Matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)  # Pause to allow the frame to render

        frame_idx += 1  # Increment frame index

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Display the final frame