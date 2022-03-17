import numpy as np
from histogram import histogram
import sys
# appending a path
sys.path.append('utils')
from utils import *

def Global_thresholding(img,threshold_value):
    Row,Column = img.shape
    thresholding_image = np.zeros((Row,Column))
    for i in range(Row):
        for j in range(Column):
            thresholding_image[i,j] = 255 if img[i,j] >= threshold_value else 0
    return thresholding_image  

def local_thresholding(image,brightness_percent):
    r, c = image.shape
    img = np.zeros_like(image, dtype=np.uint64)
    for row in range(r):    
        for col in range(c):
            img[row,col] = image[0:row,0:col].sum()
    thresholding_image = np.zeros_like(image)    

    for row in range(r):
        for col in range(c):
            count = (min(row+(c//20), r-1)-max(row-(c//20), 0))*(min(col+(c//20), c-1)-max(col-(c//20), 0))
            sum_ = img[min(row+(c//20), r-1), min(col+(c//20), c-1)]-img[max(row-(c//20), 0), min(col+(c//20), c-1)]-img[min(row+(c//20), r-1), max(col-(c//20), 0)]+img[max(row-(c//20), 0), max(col-(c//20), 0)]
            thresholding_image[row, col] = 0 if image[row, col]*count < sum_*(100.-brightness_percent)/100. else 255

    return thresholding_image


