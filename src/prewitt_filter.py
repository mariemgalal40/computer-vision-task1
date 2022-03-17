import cv2 as cv
import numpy as np
import sys

# appending a path
sys.path.append('utils')
from utils import generic_convolve, convert_to_grayScale

from utils import generic_convolve

def __construct_kernel(kernel_size):
    # Kernel construction
    mid_index = kernel_size//2
    kernel_row = [i for i in range(kernel_size)]
    
    for i in range(kernel_size):
        if i < mid_index:
            kernel_row[i] = -1
        elif i == mid_index:
            kernel_row[i] = 0
        else:
            kernel_row[i] = 1
    kernel = [kernel_row for i in range(kernel_size)]
    kernel = np.array(kernel)
    
    return kernel
    

def PrewittFilter(img: np.ndarray, kernel_size: int, kernel_type="y") -> np.ndarray :
    
    img = convert_to_grayScale(img)
    #    Margin is the unused area of image
    #    ex: kernel 3x3 makes a margin of 2 rows and 2 cols in the img
    margin = (kernel_size//2) * 2
    
    # create the prewitt kernel
    kernel = np.ones((kernel_size,kernel_size), dtype=np.uint8)
    
    
    kernel = __construct_kernel(kernel_size)
    
    if kernel_type == "y":
        pass
        
    elif kernel_type == "x":
        kernel = kernel.transpose()
        
    else:
        raise TypeError("kernel_type should take 'x' or 'y' ONLY")
            
    return generic_convolve(img, kernel)




# img = cv.imread(r"E:\images/ana w abaas.jpeg", cv.IMREAD_GRAYSCALE)
# prewitt = PrewittFilter(img, 5)
# cv.imwrite('prewitt with convolve and construct2.png', prewitt)