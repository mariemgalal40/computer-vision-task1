import cv2
import numpy as np
from matplotlib import pyplot as plt


## gaussian filter
def distance (x,y):
    return np.sqrt((x[0]-y[0])**2 +(x[1]-y[1])**2)

def gaussianLPF(D0,image):
    imgshape=image.shape
    base=np.zeros(imgshape[:2])
    row ,collunm =imgshape[:2]
    center=(row/2 ,collunm/2)
    for x in range (collunm):
        for y in range(row):
            base[y,x]=np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base
    
def gaussianHPF(D0,image):
    imgshape=image.shape
    base=np.zeros(imgshape[:2])
    row ,collunm =imgshape[:2]
    center=(row/2 ,collunm/2)
    for x in range (collunm):
        for y in range(row):
            base[y,x]=1-np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
        
        
    return base


def LPF_HPF_HYPRID(img1,img2,D0):
     # convert the images to frequence domain using np
    # we will take the origin from corner to centre  that will see tha white spot LFC
    try:
        img_fr_or1 = np.fft.fft2(img1)
        img_fr_ce1 = np.fft.fftshift(img_fr_or1)


        img_fr_or2 = np.fft.fft2(img2)
        img_fr_ce2 = np.fft.fftshift(img_fr_or2)
        

        Low_pass_ce= img_fr_ce1 * gaussianLPF(D0,img1)
        low_pass_or=np.fft.ifftshift(Low_pass_ce)
        low_spe_frin=np.fft.ifft2(low_pass_or)
        img_af_LPF=np.abs(low_spe_frin)

        high_pass_ce= img_fr_ce2 * gaussianHPF(D0,img2)
        high_pass_or=np.fft.ifftshift(high_pass_ce)
        high_spe_frin=np.fft.ifft2(high_pass_or)
        img_af_HPF=np.abs(high_spe_frin)


        hyperimage= np.abs((np.add(high_spe_frin,low_spe_frin)))
        
        return img_af_LPF , img_af_HPF , hyperimage
    except:
        print('image should be in same size')









