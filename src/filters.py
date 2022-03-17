import numpy as np
import sys
sys.path.append('utils')
from utils import convolve

def average_filter(image):
    mask = np.ones([3,3], dtype = int)
    mask = mask/9
    return convolve(image, mask)

def gaussian_mask(size, sigma):
     ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
     xx, yy = np.meshgrid(ax, ax)
     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
     return kernel / np.sum(kernel)

def gaussian_fitler(image):
    return convolve(image, gaussian_mask(3,2))

def median_filter(image):
    rows, columns = image.shape
    img_new = np.zeros([rows, columns], dtype=np.uint8)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            temp = [image[i-1, j-1],
                    image[i-1, j],
                    image[i-1, j + 1],
                    image[i, j-1],
                    image[i, j],
                    image[i, j + 1],
                    image[i + 1, j-1],
                    image[i + 1, j],
                    image[i + 1, j + 1]]
            temp = sorted(temp)
            img_new[i][j] = temp[4]
    img_new = img_new.astype(np.uint8)
    return img_new