import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pathlib
import cv2
import numpy as np

# Download and extract the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

# Set parameters
batch_size = 32
img_height = 224
img_width = 224

# Prepare the data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# validation_generator = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # Load the MobileNetV2 model, excluding the top layer
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# # Add new top layers for our specific task
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Number of classes

# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze the base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Fine-tune the model
# model.fit(train_generator, epochs=10, validation_data=validation_generator, steps_per_epoch=len(train_generator), validation_steps=len(validation_generator))

# # Save the model
# model.save('flower_model.h5')

# Load the trained model
model = tf.keras.models.load_model('flower_model.h5')

# Define the class labels
class_labels = list(train_generator.class_indices.keys())

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Check if the camera can be opened
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Unable to open camera with index {camera_index}")
else:
    print(f"Camera with index {camera_index} opened successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make predictions
        predictions = model.predict(processed_frame)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the results
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Flower Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
