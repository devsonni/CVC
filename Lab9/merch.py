import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, InceptionV3
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Load the MerchData dataset
data_dir = 'MerchData'

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

def build_model(base_model, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    if base_model.name == "inception_v3":
        x = GlobalAveragePooling2D()(base_model.output)
    else:
        x = Flatten()(base_model.output)

    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# Load the pretrained VGG19 model and add custom layers
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg19_model = build_model(vgg19_base, train_generator.num_classes)

# Load the pretrained InceptionV3 model and add custom layers
inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_model = build_model(inception_base, train_generator.num_classes)

# Compile the models
vgg19_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
inception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the VGG19 model
vgg19_history = vgg19_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Print the keys in the history dictionary for debugging
print("Keys in VGG19 history.history:", vgg19_history.history.keys())

# Train the InceptionV3 model
inception_history = inception_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Print the keys in the history dictionary for debugging
print("Keys in InceptionV3 history.history:", inception_history.history.keys())

# Plot training loss and validation accuracies for VGG19
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(vgg19_history.history['accuracy'], 'r', label='Training accuracy')
if 'val_accuracy' in vgg19_history.history:
    plt.plot(vgg19_history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.title('VGG19 Training and validation accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(vgg19_history.history['loss'], 'r', label='Training loss')
if 'val_loss' in vgg19_history.history:
    plt.plot(vgg19_history.history['val_loss'], 'b', label='Validation loss')
plt.title('VGG19 Training and validation loss')
plt.legend()

# Plot training loss and validation accuracies for InceptionV3
plt.subplot(2, 2, 3)
plt.plot(inception_history.history['accuracy'], 'r', label='Training accuracy')
if 'val_accuracy' in inception_history.history:
    plt.plot(inception_history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.title('InceptionV3 Training and validation accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(inception_history.history['loss'], 'r', label='Training loss')
if 'val_loss' in inception_history.history:
    plt.plot(inception_history.history['val_loss'], 'b', label='Validation loss')
plt.title('InceptionV3 Training and validation loss')
plt.legend()

plt.tight_layout()
plt.show()

# Test the models with random images
test_image_paths = ['Test/Cap.jpeg', 'Test/RubicCube.png', 'Test/Torch.jpeg', 'Test/ScrewDriver.jpeg']

for image_path in test_image_paths:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    vgg19_prediction = vgg19_model.predict(img_array)
    inception_prediction = inception_model.predict(img_array)

    vgg19_predicted_class = np.argmax(vgg19_prediction[0])
    inception_predicted_class = np.argmax(inception_prediction[0])

    class_labels = list(train_generator.class_indices.keys())

    print(f"VGG19 predicted class for {image_path}: {class_labels[vgg19_predicted_class]}")
    print(f"InceptionV3 predicted class for {image_path}: {class_labels[inception_predicted_class]}")
