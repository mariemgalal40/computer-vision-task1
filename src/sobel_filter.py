import cv2 as cv
import numpy as np
import sys
# appending a path
sys.path.append('utils')

from utils import generic_convolve, convert_to_grayScale


def __construct_kernel(kernel_size):
    # Kernel construction
    
    start_value = kernel_size//2
    mid_index = kernel_size//2
    kernel_row = [0 for i in range(kernel_size)]
    middle_row = [2*i for i in kernel_row]

    for i in range(kernel_size):
        kernel_row[i] = start_value
        start_value -= 1

    kernel = [kernel_row for i in range(kernel_size)]
    kernel[mid_index] = middle_row
    kernel = np.array(kernel)
    return kernel


def SobelFilter(img: np.ndarray, kernel_size: int, kernel_type="y") -> np.ndarray :
    
    img = convert_to_grayScale(img)
    #    Margin is the unused area of image
    #    ex: kernel 3x3 makes a margin of 2 rows and 2 cols in the img
    margin = (kernel_size//2) * 2
    
    # create the Sobel kernel
    #kernel = np.ones(kernel_size, dtype=np.uint8)
    
    kernel = __construct_kernel(kernel_size)
    
    if kernel_type == "y":
        pass
        
    elif kernel_type == "x":
        kernel = kernel.transpose()
        
    else:
        raise TypeError("kernel_type should take 'x' or 'y' ONLY")
    
    return generic_convolve(img, kernel)






# img = cv.imread(r"E:\images/ana w abaas.jpeg", cv.IMREAD_GRAYSCALE)
# sobel = SobelFilter(img,5)
# cv.imwrite('sobel after all modifications.png', sobel)