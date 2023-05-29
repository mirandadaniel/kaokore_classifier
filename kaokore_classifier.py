import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def parse_label(label):
    label_parts = label.split(',')
    class_value = int(label_parts[1])

    return class_value

def folder_to_pixel_array(folder_path, target_size=(256, 256)):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    pixel_array = []

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path).resize(target_size)
        pixel_data = np.array(image)
        pixel_array.append(pixel_data)
        image.close()
    pixel_array = np.array(pixel_array)

    return pixel_array

folder_path = "/home/daniel/kaokore/kaokore/images_256_3/"
pixel_array = folder_to_pixel_array(folder_path)
width = 256
height = 256

first_image_pixels = pixel_array[345]
print(first_image_pixels)

labels_file = "/home/daniel/kaokore/kaokore/labels_2.csv"
with open(labels_file, 'r') as file:
    labels = file.read().splitlines()
    labels.pop(0)
    print("labels type is: ")
    print(labels)

if len(pixel_array) != len(labels):
    raise ValueError("Number of images and labels don't match.")

X_train, X_test, y_train, y_test = train_test_split(pixel_array, labels, test_size=0.2, random_state=42)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

num_classes = len(set(labels))
y_train = [parse_label(label) for label in y_train]
y_test = [parse_label(label) for label in y_test]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
