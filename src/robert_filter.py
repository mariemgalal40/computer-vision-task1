import cv2 as cv
import numpy as np
import sys
# appending a path
sys.path.append('utils')

from utils import generic_convolve, convert_to_grayScale



def __construct_kernel(kernel_type):
    # Kernel construction
    if kernel_type == 'x':
        kernel = [ [1,  0],
                   [0, -1]  ]
    elif kernel_type == 'y':
        kernel = [ [0, -1],
                   [1,  0]  ]
    else:
        raise Exception("Invalid Kernel Type")
        
    return np.array(kernel, dtype=np.uint8)
    
    

def RobertFilter(img: np.ndarray, kernel_size: int, kernel_type="y") -> np.ndarray :
        
    img = convert_to_grayScale(img)
    #    Margin is the unused area of image
    #    ex: kernel 3x3 makes a margin of 2 rows and 2 cols in the img
    margin = (kernel_size//2) * 2
    
    # create the Sobel kernel
    #kernel = np.ones(kernel_size, dtype=np.uint8)
    
    kernel = __construct_kernel(kernel_type)
    
    return generic_convolve(img, kernel)
    
    

# img = cv.imread(r"E:\images/ana w abaas.jpeg", cv.IMREAD_GRAYSCALE)
# robert = RobertFilter(img,2, kernel_type='y')
# negative = negativeTransfrom(robert)
# cv.imwrite('robert after all modificationsY.png', negative)