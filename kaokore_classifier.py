# import os
# from PIL import Image
# import numpy as np
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def parse_label(label):
    # Split the label by comma
    label_parts = label.split(',')

    # Extract the class (assuming it's the second element)
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

    # Convert the list of pixel arrays to a numpy array
    pixel_array = np.array(pixel_array)

    return pixel_array


    # Load the first image to get the size
    # image_path = os.path.join(folder_path, image_files[0])
    # image = Image.open(image_path)
    # width, height = image.size
    # image.close()

    # # Create an empty array to store the flattened pixel data
    # pixel_array = np.zeros((len(image_files), width * height, 3), dtype=np.uint8)

    # # Iterate over the images and convert them to pixel arrays
    # for i, file_name in enumerate(image_files):
    #     image_path = os.path.join(folder_path, file_name)
    #     image = Image.open(image_path)
    #     pixel_data = np.array(image)
    #     pixel_array[i] = pixel_data.reshape(-1, 3)
    #     image.close()

    # return pixel_array, width, height

# Example usage
folder_path = "/home/daniel/kaokore/kaokore/images_256_3/"
pixel_array = folder_to_pixel_array(folder_path)
# pixel_array = np.array(pixel_array)
# pixel_array, width, height = folder_to_pixel_array(folder_path)
width = 256
height = 256

# Access pixel data for the first image
first_image_pixels = pixel_array[345]
print(first_image_pixels)

print("pixel array se is: ", len(pixel_array))

# Load labels (assuming labels are stored in a separate file)

# The og: 
# labels_file = "/home/daniel/kaokore/kaokore/labels.csv" 

# labels_file = "/home/daniel/kaokore/dataset/labels.csv"
labels_file = "/home/daniel/kaokore/kaokore/labels_2.csv"
with open(labels_file, 'r') as file:
    labels = file.read().splitlines()
    labels.pop(0)
    print("labels type is: ")
    print(labels)


# Split the data into training and testing sets
# print("pixel array?", pixel_array)

print("pixel array length: ", len(pixel_array))
print("lenght of labels: ", len(labels))
if len(pixel_array) != len(labels):
    raise ValueError("Number of images and labels don't match.")

X_train, X_test, y_train, y_test = train_test_split(pixel_array, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be in the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
num_classes = len(set(labels))
y_train = [parse_label(label) for label in y_train]
y_test = [parse_label(label) for label in y_test]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# width, height = image.size

# Build the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)