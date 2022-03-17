import numpy as np
import random
from utils import ceil_floor_image

def uniform_noise(image, low=-68, high=68):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uniform_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    output = image + uniform_noise
    output = ceil_floor_image(output)
    return output

def gaussian_noise(image, mean=0, std=127):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaussian_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    output = image + gaussian_noise
    output = ceil_floor_image(output)
    return output

def salt_pepper_noise(image, prop=0.05):
    rows, columns = image.shape
    output = np.zeros(image.shape, np.uint8)
    for i in range(rows):
        for j in range(columns):
            r = random.random()
            if r < prop/2:
                #pepper sprinkled
                output[i][j] = 0
            elif r < prop:
                #salt sprinkled
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output