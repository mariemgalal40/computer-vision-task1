import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from histogram import *


def split_color_channels(img):
    '''
    Take img split its color channel and return them in order
    '''
    first = img[:,:,0]
    second = img[:,:,1]
    third = img[:,:,2]
    
    return first, second, third


def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])                

def plotdist(distr):
    plt.plot(distr)  
    plt.show()

def plothist(histog):
    plt.bar(list(range(0, 256)), histog)
    plt.show()


def generic_convolve(mat, kernel):
    '''
        Make convolution operation between a gray scale img and nxn kernel
    '''
    margin = (kernel.shape[0]//2) * 2
    # convolution
    img_rows = mat.shape[0]
    img_cols = mat.shape[1]

    # resultant img
    filtered_img = np.ones(mat.shape)
    
    for row in range(img_rows-margin):
        for col in range(img_cols-margin):

            # convolution operation
            result = 0
            for K_row in range(kernel.shape[0]):
                for K_col in range(kernel.shape[1]):
                    result += mat[row+K_row][col+K_col] * kernel[K_row][K_col]

            filtered_img[row+1][col+1] = result
            
    return filtered_img

def convert_to_grayScale(img):
    """
        Convert color image to gray image
    """
    if len(list(img.shape)) == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    elif len(list(img.shape)) == 2: pass
    else:
        raise Exception("Invalid image dimensions")
    return img
    

#function that takes an image and return image in uint8 type and pixels in range 0:255
def ceil_floor_image(image):
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

#function for convolution that takes the image and the mask
def convolve(image, mask):
    rows, columns, channels = image.shape
    img_new = np.zeros([rows, columns, channels], dtype=np.uint8)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            temp = image[i-1, j-1]*mask[0, 0]+image[i-1, j]*mask[0, 1]+image[i-1, j + 1]*mask[0, 2]+image[i, j-1]*mask[1, 0]+ image[i, j]*mask[1, 1]+image[i, j + 1]*mask[1, 2]+image[i + 1, j-1]*mask[2, 0]+image[i + 1, j]*mask[2, 1]+image[i + 1, j + 1]*mask[2, 2]
            img_new[i, j]= temp
    img_new = img_new.astype(np.uint8)
    return img_new
    
    
#function to plot or show the images
def imshow_multi(imgs, labels):
    _, axs = plt.subplots(5, 4, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax, label in zip(imgs, axs, labels):
        ax.imshow(img)
        ax.set_title(label)
    plt.show()
    

#function to plot or show the images
def generic_imshow_multi(imgs, labels, nrow, ncol):
    _, axs = plt.subplots(ncol, nrow, figsize=(18, 18))
    axs = axs.flatten()
    for img, ax, label in zip(imgs, axs, labels):
        ax.imshow(img, cmap='gray')
        ax.set_title(label)
    plt.show()

    
def negativeTransfrom(img):
    # transform = 255-x
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img[row][col] = 255 - img[row][col]
            
    return img


    
    
        
      
