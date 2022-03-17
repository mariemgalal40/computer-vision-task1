import numpy as np
import matplotlib.pyplot as plt

def histogram(image):
    bins = np.zeros(256, np.int32)
    row,column=image.shape
    for i in range(row):
        for j in range(column):
            bins[image[i,j]] += 1
    histo = np.squeeze(bins) 
    return histo  
    
