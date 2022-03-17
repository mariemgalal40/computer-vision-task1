import numpy as np
from histogram import histogram
import sys
# appending a path
sys.path.append('utils')
from utils import *



def cdf(histogram):  # cumulative distribution frequency
    cdf = [0] * 256   
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i]= cdf[i-1]+histogram[i]
    cdf = [sum(histogram[:i+1]) for i in range(256)]
    cdf /= cdf[-1]
    return cdf     
def image_equalizing(image):
    trans_map = np.floor(255 * cdf(histogram(image)))
    img1d = list(image.flatten())
    map_img1d = [trans_map[px] for px in img1d]
    equalizedimage = np.reshape(np.asarray(map_img1d), image.shape)
    return equalizedimage
