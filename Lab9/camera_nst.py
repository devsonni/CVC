import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess the style image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Load the style image
style_image_path = 'art.jpg'
style_image = load_image(style_image_path)

# Open the video stream
cap = cv2.VideoCapture(1)

# Define the desired width and height for resizing
resize_width, resize_height = 200, 200

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the desired dimensions
    small_frame = frame #cv2.resize(frame, (resize_width, resize_height))

    # Convert the frame to a tensor
    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame_tensor = tf.convert_to_tensor(small_frame_rgb, dtype=tf.float32)
    small_frame_tensor = small_frame_tensor[tf.newaxis, ...]

    # Apply the style transfer
    stylized_image = model(tf.constant(small_frame_tensor), tf.constant(style_image))[0]

    # Convert the stylized image to a format suitable for OpenCV
    stylized_image = np.squeeze(stylized_image)
    stylized_image = np.clip(stylized_image * 255, 0, 255).astype(np.uint8)
    stylized_image_bgr = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)

    # Resize the stylized image back to the original frame size
    output_frame = cv2.resize(stylized_image_bgr, (frame.shape[1], frame.shape[0]))

    # Display the stylized image
    cv2.imshow('Stylized Video Feed', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
