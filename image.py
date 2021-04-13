from imageio import imread
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path):
    """ Read an image and return a one hot vector of the image"""
    img = imread(image_path)
    reshape_value = 1

    for i in img.shape:
        reshape_value *= i

    return img.reshape((1, reshape_value)), img.shape


def show_image(image):
    """ Show a single image"""
    plt.imshow(image)
    plt.show()


def show_images(a, b):
    """ Show two images side by side"""
    plot_image = np.concatenate((a, b), axis=1)
    plt.imshow(plot_image)
    plt.show()
