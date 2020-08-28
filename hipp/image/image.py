import PIL
import os
import cv2
import numpy as np
from skimage import exposure

import hipp

"""
Library for image processing functions. 
"""
    
def clahe_equalize_image(img_gray,
                         clipLimit = 2.0,
                         tileGridSize = (8,8)):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_gray_clahe = clahe.apply(img_gray)
    return img_gray_clahe
    
def img_linear_stretch(img_gray,
                       min_max = (0.1, 99.9)):
    p_min, p_max = np.percentile(img_gray, min_max)
    img_rescale = exposure.rescale_intensity(img_gray, in_range=(p_min, p_max))
    return img_rescale
    
def threshold_and_add_noise(image_array,
                            threshold=50):
    mask = image_array > threshold
    rand = np.random.randint(0,256,size=image_array.shape)
    image_array[mask] = rand[mask]
    return image_array