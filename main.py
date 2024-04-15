import string
import random

random.seed(353)
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
from collections import Counter
from matplotlib import pyplot as plt
import math
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from ipywidgets import interact
import ipywidgets as ipywidgets
from keras.preprocessing.image import ImageDataGenerator


def generate_image_tuples(parent_folder):
    """
    Generate a list of tuples where each tuple's first element is the folder name
    and the second element is one image read from that folder.

    Parameters: parent_dir: Path to the parent folder
    Returns: List of tuples [(folder_name, image1), (folder_name, image2), ...]
    """

    image_tuples = []

    # Iterate over each item in the parent_folder
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)

                # # Open the JPG file in binary mode
                # with open(image_path, 'rb') as f:
                #     # Read the binary data
                #     binary_data = f.read()
                #
                cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                print(cv_image.shape)
                image_tuples.append((folder_name, cv_image))
        # else print:
        else:
            print("this folder doesn't have subfolders, use files_in_folder function")

    # image_tuples.sort()
    # cannot simply sort since many items share the same first letter/number

    return image_tuples


# map the character to int by their unicode code representation
# A - Z as 0 - 25 and 0 - 9 as 26 - 35
def char_to_int(char):
    if 'A' <= char <= 'Z':
        return ord(char) - ord('A')
    elif '0' <= char <= '9':
        return 26 + ord(char) - ord('0')
    elif char == '_':
        return 36
    else:
        raise ValueError(f"Invalid character: {char}")


NUMBER_OF_LABELS = 37
CONFIDENCE_THRESHOLD = 0.01


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


if __name__ == '__main__':
    image_folder_path = "/home/fizzer/PycharmProjects/character_cnn/image/my_controller_image/labelled"
    images = generate_image_tuples(image_folder_path)

    np.random.shuffle(images)
    # print(len(images))

    # print(images[0:2])
    # Generate X and Y datasets
    X_dataset_orig = np.array([data[1] for data in images])
    Y_dataset_orig = np.array([data[0] for data in images])
    Y_dataset_orig = np.array([char_to_int(char) for char in Y_dataset_orig])

    print(X_dataset_orig.shape)
    print(Y_dataset_orig.shape)

    print(X_dataset_orig[0, 40, 40])

    print(Y_dataset_orig[230:250])

    # Normalize X (images) dataset
    X_dataset = X_dataset_orig / 255.  # if not normalized, learning will be slow
    # Convert Y dataset to one-hot encoding
    Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS)
    print(Y_dataset[0])

    VALIDATION_SPLIT = 0.09

    split_index = math.ceil(X_dataset.shape[0] * (1 - VALIDATION_SPLIT))

    X_resized = np.empty((X_dataset.shape[0], 120, 60), dtype=np.float64)
    for i, image in enumerate(X_dataset):
        resized_image = cv2.resize(image, (60, 120)).astype(
            np.float64)  # Resize each image to (120, 60) and convert to float64
        X_resized[i] = resized_image

    X_train_dataset = X_resized[:split_index]
    Y_train_dataset = Y_dataset[:split_index]

    X_val_dataset = X_resized[split_index:]
    Y_val_dataset = Y_dataset[split_index:]

    print("X shape: " + str(X_resized.shape))
    print("Y shape: " + str(Y_dataset.shape))
    print("Total examples: {:d}\nTraining examples: {:d}\n"
          "Validation examples: {:d}".
          format(X_resized.shape[0],
                 X_train_dataset.shape[0],
                 X_val_dataset.shape[0]))

    # Convert the list to a NumPy array
    X_dataset = np.array(X_resized)


    # # Display images in the training data set.
    # def displayImage(index):
    #     plt.imshow(X_dataset[index])
    #     caption = ("y = " + str(Y_dataset[index]))  # str(np.squeeze(Y_dataset_orig[:, index])))
    #     plt.text(0.5, 0.5, caption,
    #              color='orange', fontsize=20,
    #              horizontalalignment='left', verticalalignment='top')
    #
    #
    # interact(displayImage,
    #          index=ipywidgets.IntSlider(min=0, max=X_dataset_orig.shape[0],
    #                                     step=1, value=10))


    def reset_weights(model):
        for ix, layer in enumerate(model.layers):
            if (hasattr(model.layers[ix], 'kernel_initializer') and
                    hasattr(model.layers[ix], 'bias_initializer')):
                weight_initializer = model.layers[ix].kernel_initializer
                bias_initializer = model.layers[ix].bias_initializer

                old_weights, old_biases = model.layers[ix].get_weights()

                model.layers[ix].set_weights([
                    weight_initializer(shape=old_weights.shape),
                    bias_initializer(shape=len(old_biases))])

    def build_model():
        inputLayer = tf.keras.layers.Input(shape=(120, 60, 1))
        conv_model = tf.keras.layers.Rescaling(1./255)(inputLayer)

        conv_model = tf.keras.layers.Conv2D(32, 3, padding='same')(conv_model)
        conv_model = tf.keras.layers.BatchNormalization()(conv_model)
        conv_model = tf.keras.layers.Activation('relu')(conv_model)
        conv_model = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv_model)

        conv_model = tf.keras.layers.Conv2D(32, 3, padding='same')(conv_model)
        conv_model = tf.keras.layers.BatchNormalization()(conv_model)
        conv_model = tf.keras.layers.Activation('relu')(conv_model)
        conv_model = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv_model)

        conv_model = tf.keras.layers.Conv2D(32, 3, padding='same')(conv_model)
        conv_model = tf.keras.layers.BatchNormalization()(conv_model)
        conv_model = tf.keras.layers.Activation('relu')(conv_model)
        conv_model = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2)(conv_model)

        conv_model = tf.keras.layers.Flatten()(conv_model)
        conv_model = tf.keras.layers.Dense(64)(conv_model)
        conv_model = tf.keras.layers.BatchNormalization()(conv_model)
        conv_model = tf.keras.layers.Dense(37, activation='softmax')(conv_model)
        return tf.keras.models.Model(inputs=inputLayer, outputs=conv_model)

    # remove layers, neurons
    # reduce input size

    conv_model = build_model()
    # conv_model = models.Sequential()
    #
    # conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
    #                              input_shape=(80, 45, 1)))  # dimension of x dataset
    # conv_model.add(layers.BatchNormalization())
    # conv_model.add(layers.MaxPooling2D((2, 2)))
    #
    # conv_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # conv_model.add(layers.BatchNormalization())
    # conv_model.add(layers.MaxPooling2D((2, 2)))
    #
    # # conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # # conv_model.add(layers.BatchNormalization())
    # # conv_model.add(layers.MaxPooling2D((2, 2)))
    #
    # # conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # # conv_model.add(layers.BatchNormalization())
    # # conv_model.add(layers.MaxPooling2D((2, 2)))
    #
    # conv_model.add(layers.Flatten())
    # conv_model.add(layers.Dense(128, activation='relu'))
    # conv_model.add(layers.Dropout(0.5))
    # # conv_model.add(layers.Dense(64, activation='relu'))
    # conv_model.add(layers.Dense(37, activation='softmax'))
    conv_model.summary()

    LEARNING_RATE = 1e-3
    conv_model.compile(loss='categorical_crossentropy',
                       optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE),
                       metrics=['acc'])
    reset_weights(conv_model)

    history_conv = conv_model.fit(X_resized, Y_dataset,
                                  epochs=84,
                                  batch_size=64, validation_split=VALIDATION_SPLIT)

    # Plot model loss
    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], frameon=False)
    plt.savefig("model loss.png")
    plt.show()

    # Plot model accuracy
    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], frameon=False)
    plt.savefig("model accuracy.png")
    plt.show()

    conv_model.save('character_prediction_local_0415.h5')