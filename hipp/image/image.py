import PIL
import os
import cv2
import numpy as np
from skimage import exposure
from skimage import transform as tf

import hipp

"""
Library for image processing functions. 
"""

def affine_transform_image(image_array, 
                           coordinates, 
                           coordinates_true):
    
    output_dim = image_array.shape
    
    tform = tf.AffineTransform()
    tform.estimate(coordinates, coordinates_true)
    
    transformed = tform(coordinates)
    
    image_array_transformed = tf.warp(image_array, tform, output_shape=output_dim)
    image_array_transformed = (image_array_transformed*255).astype(np.uint8)
    
    return image_array_transformed, tform
        
def clahe_equalize_image(img_gray,
                         clipLimit = 2.0,
                         tileGridSize = (8,8)):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_gray_clahe = clahe.apply(img_gray)
    return img_gray_clahe

def crop_about_point(image_array,
                     point_yx,
                     output_shape = 11250):
    
    distance_from_point = int(output_shape/2)
    x_L = point_yx[1]-distance_from_point
    x_R = point_yx[1]+distance_from_point
    y_T = point_yx[0]-distance_from_point
    y_B = point_yx[0]+distance_from_point
    
    cropped_array = image_array[y_T:y_B, x_L:x_R]
    
    return cropped_array
    
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