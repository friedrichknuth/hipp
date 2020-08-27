import cv2
import glob
import numpy as np
import os
import pandas as pd
import pathlib
import shutil

import hipp

"""
Library with core image pre-processing functions.
"""

def create_fiducial_template(image_file, 
                             output_directory = 'fiducials',
                             output_file_name='fiducial.tif',
                             distance_around_fiducial=100):
     
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    df = hipp.tools.point_picker(image_file)

    fiducial = (df.x[0],df.y[0])
    
    x_L = int(fiducial[0]-distance_around_fiducial)
    x_R = int(fiducial[0]+distance_around_fiducial)
    y_T = int(fiducial[1]-distance_around_fiducial)
    y_B = int(fiducial[1]+distance_around_fiducial)
    cropped = image_array[y_T:y_B, x_L:x_R]
    
    out = os.path.join(output_directory,output_file_name)
    cv2.imwrite(out,cropped)
    
    return out
    
    
def crop_fiducial(image_file,
                  image_array,
                  match,
                  label=None,
                  distance_from_loc = 200,
                  output_directory='tmp/fiducial_crop'):
    
    x_L = match[1]
    x_R = match[1]+distance_from_loc
    y_T = match[0]
    y_B = match[0]+distance_from_loc
    fiducial_crop_array = image_array[y_T:y_B, x_L:x_R]

    file_path, file_name, file_extension = hipp.io.split_file(image_file)

    output_file_name = os.path.join(output_directory,file_name+'_'+label+file_extension)

    cv2.imwrite(output_file_name,fiducial_crop_array)
    
    return output_file_name


def define_center_windows(image_array):
    
    half_image_height     = int(image_array.shape[0] / 2)
    quarter_image_height  = int(half_image_height / 2)

    half_image_width     = int(image_array.shape[1] / 2)
    quarter_image_width  = int(half_image_width / 2)
    
    center_left    = [quarter_image_height,
                      half_image_height + quarter_image_height,
                      0, 
                      quarter_image_width]

    center_top     = [0,
                      quarter_image_height,
                      quarter_image_width,
                      half_image_width + quarter_image_width]

    center_right   = [quarter_image_height,
                      half_image_height + quarter_image_height,
                      half_image_width + quarter_image_width,
                      image_array.shape[1]]


    center_bottom  = [half_image_height + quarter_image_height,
                      image_array.shape[0],
                      quarter_image_width,
                      half_image_width + quarter_image_width]
                     
    center_windows = [center_left, center_top, center_right, center_bottom]
    
    return center_windows

def define_corner_windows(image_array):
    
    half_image_height     = int(image_array.shape[0] / 2)
    quarter_image_height  = int(half_image_height / 2)

    half_image_width     = int(image_array.shape[1] / 2)
    quarter_image_width  = int(half_image_width / 2)
    
    corner_top_left     = [0,
                           quarter_image_height,
                           0, 
                           quarter_image_width]

    corner_top_right    = [0,
                           quarter_image_height,
                           half_image_width + quarter_image_width,
                           image_array.shape[1]]

    corner_bottom_right = [half_image_height + quarter_image_height,
                           image_array.shape[0],
                           half_image_width + quarter_image_width,
                           image_array.shape[1]]


    corner_bottom_left  = [half_image_height + quarter_image_height,
                           image_array.shape[0],
                           0,
                           quarter_image_width]
                     
    corner_windows = [corner_top_left, corner_top_right, corner_bottom_right, corner_bottom_left]
    
    return corner_windows

def detect_fiducials(slices,
                     template_array,
                     windows):

    matches = []

    for index, slice_array in enumerate(slices):
        match = hipp.core.match_template(slice_array,
                                         template_array)
                                         
        match = (windows[index][0] + match[0],
                 windows[index][2] + match[1])
        matches.append(match)
        
    return matches


def detect_high_res_fiducial(fiducial_crop_high_res_file,
                             template_high_res_zoomed_file,
                             distance_from_loc=200,
                             qc=True):
    
    fiducial_crop_high_res_array = cv2.imread(fiducial_crop_high_res_file,cv2.IMREAD_GRAYSCALE)
    template_high_res_zoomed_array = cv2.imread(template_high_res_zoomed_file,cv2.IMREAD_GRAYSCALE)
    
    match_location = hipp.core.match_template(fiducial_crop_high_res_array,
                                              template_high_res_zoomed_array)

    if qc == True:
        output_directory = 'qc/fiducial_detection/'
        p = pathlib.Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)
        
        file_path, file_name, file_extension = hipp.io.split_file(fiducial_crop_high_res_file)
        
        image_array = cv2.cvtColor(fiducial_crop_high_res_array,cv2.COLOR_GRAY2RGB)
        image_array[(match_location[0]+int(distance_from_loc/2),match_location[1]+int(distance_from_loc/2))] = 255,0,0
        
        x_L = match_location[1]
        x_R = match_location[1]+distance_from_loc
        y_T = match_location[0]
        y_B = match_location[0]+distance_from_loc
        image_array = image_array[y_T:y_B, x_L:x_R]
        
        out = os.path.join(output_directory,file_name+file_extension)
        cv2.imwrite(out,image_array)
    
    return match_location
    

def detect_subpixel_fiducial_coordinates(image_file,
                                         image_array,
                                         matches,
                                         template_high_res_zoomed_file,
                                         labels = ['center_left', 'center_top', 'center_right', 'center_bottom'],
                                         distance_from_loc = 200,
                                         factor = 8,
                                         cleanup=True,
                                         qc=True):
    
    output_directory  ='tmp/fiducial_crop'
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    subpixel_fiducial_locations = []

    for index, match_location in enumerate(matches):

        cropped_fiducial_file = hipp.core.crop_fiducial(image_file,
                                                        image_array,
                                                        match_location,
                                                        label=labels[index],
                                                        distance_from_loc = distance_from_loc,
                                                        output_directory='tmp/fiducial_crop')

        fiducial_crop_high_res_file = hipp.utils.enhance_geotif_resolution(cropped_fiducial_file,
                                                                           factor=factor)

        match_location_high_res = hipp.core.detect_high_res_fiducial(fiducial_crop_high_res_file,
                                                                     template_high_res_zoomed_file,
                                                                     distance_from_loc=distance_from_loc,
                                                                     qc=qc)

        y,x = ((match_location_high_res[0]+int(distance_from_loc/2))/factor,
               (match_location_high_res[1]+int(distance_from_loc/2))/factor)

        subpixel_fiducial_location = y + match_location[0], x+match_location[1]

        subpixel_fiducial_locations.append(subpixel_fiducial_location)
        
    if cleanup == True:
        shutil.rmtree('tmp/')
        
    return subpixel_fiducial_locations
    

def iter_detect_fiducials(image_files_directory = 'input_data/raw_images/',
                          image_files_extension ='.tif',
                          template_file = None,
                          template_high_res_zoomed_file = None,
                          center_fiducials=False,
                          corner_fiducials=False,
                          qc=True):
    
    """
    Function to iteratively detect fiducial markers in a set of images and return as pandas.DataFrame.
                           
    Ensure that the templates correspond to either the fiducial markers at the center of the image edges, 
    or at the image corners. Specify flag accordingly.
    """
    
    images = sorted(glob.glob(os.path.join(image_files_directory,'*'+image_files_extension)))
    template_array = cv2.imread(template_file,cv2.IMREAD_GRAYSCALE)
    fiducial_locations = []
    
    for image_file in images:
        image_array = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        
        # Subset image array into window slices to speed up template matching
        if center_fiducials:
            windows = hipp.core.define_center_windows(image_array)
        elif corner_fiducials:
            windows = hipp.core.define_corner_windows(image_array)
        else:
            print("Please specify center or corner fiducials and provide corresponding templates.")
            break
        
        slices = hipp.core.slice_image_frame(image_array,windows)
        
        # Detect fiducial in each window
        matches = hipp.core.detect_fiducials(slices,
                                             template_array,
                                             windows)
        
        if center_fiducials:
            labels = ['center_left','center_top','center_right','center_bottom']
        elif corner_fiducials:
            labels = ['corner_top_left','corner_top_right','corner_bottom_right','corner_bottom_left']

        subpixel_fiducial_locations = hipp.core.detect_subpixel_fiducial_coordinates(image_file,
                                                 image_array,
                                                 matches,
                                                 template_high_res_zoomed_file,
                                                 labels = labels,
                                                 qc=qc)
                                                 
        fiducial_locations.append(subpixel_fiducial_locations)
    
    
    df1 = pd.DataFrame(images,columns=['fileName'])
    df2 = pd.DataFrame(fiducial_locations, columns =labels)
    df  = pd.concat([df1,df2], axis=1)
    
    return df


def match_template(image_array,
                   template_array):
    
    result = cv2.matchTemplate(image_array,template_array,cv2.TM_CCOEFF_NORMED)
    location = np.where(result==result.max())
    
    return location[0][0], location[1][0]
    
def slice_image_frame(image_array, 
                      windows):

    slices = []
    for window in windows:
        slice_array = image_array[window[0]:window[1],
                                   window[2]:window[3]]
        slices.append(slice_array)
        
    return slices