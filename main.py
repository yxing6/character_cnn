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


if __name__ == '__main__':
    fileBright = "/home/fizzer/PycharmProjects/character_cnn/image/my_controller_image/raw/43.jpg"
    fileDark = "/home/fizzer/PycharmProjects/character_cnn/image/my_controller_image/raw/6.jpg"

    bright = cv2.imread(fileBright)
    dark = cv2.imread(fileDark)

    cv2.imshow('RGB Bright Image', bright)
    cv2.imshow("RGB dark Image", dark)

    gray_b = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grey Bright Image', gray_b)
    cv2.imshow("Grey dark Image", gray_d)

    gray_b_brightness = np.mean(gray_b)
    gray_d_brightness = np.mean(gray_d)

    target = 2
    delta_b_brightness = target - gray_b_brightness
    delta_d_brightness = target - gray_d_brightness

    adjusted_b_image = cv2.add(gray_b, np.full_like(gray_b, delta_b_brightness, dtype=np.uint8))
    adjusted_d_image = cv2.add(gray_d, np.full_like(gray_d, delta_d_brightness, dtype=np.uint8))

    cv2.imshow('Grey Bright Image adjusted', adjusted_b_image)
    cv2.imshow("Grey dark Image adjusted", adjusted_d_image)

    threshold = 240
    _, binary_b = cv2.threshold(adjusted_b_image, threshold, 255, cv2.THRESH_BINARY)
    _, binary_d = cv2.threshold(adjusted_d_image, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Bright binary', binary_b)
    cv2.imshow("Dark binary", binary_d)

    cv2.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
