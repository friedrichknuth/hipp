import cv2
from collections.abc import Iterable
import concurrent
import glob
import numpy as np
import os
import pandas as pd
import pathlib
import psutil
import shutil
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import rasterio

import hipp.core
import hipp.image
import hipp.io
import hipp.math
import hipp.qc
import hipp.tools
import hipp.utils

"""
Library with core image pre-processing functions.
"""
    

def compute_principal_point(subpixel_fiducial_locations, 
                            subpixel_quality_scores,
                            median_scores):
    
    principal_point_estimates = []
    
    A0, B0, A1, B1 = subpixel_fiducial_locations
    A0_score, B0_score, A1_score, B1_score = subpixel_quality_scores
    A0_median_score, B0_median_score, A1_median_score, B1_median_score = median_scores
    
    principal_point_A = hipp.core.eval_and_compute_principal_point(A0,A0_score,A0_median_score,
                                                                   A1,A1_score,A1_median_score)
    if principal_point_A:
        principal_point_estimates.append(principal_point_A)

    principal_point_B = hipp.core.eval_and_compute_principal_point(B0,B0_score,B0_median_score,
                                                                   B1,B1_score,B1_median_score)
    if principal_point_B:
        principal_point_estimates.append(principal_point_B)
        
    principal_point_estimates = np.array(principal_point_estimates)
    if principal_point_estimates.size != 0:
        principal_point = principal_point_estimates[:,0].mean(), principal_point_estimates[:,1].mean()
        return principal_point
    

def compute_principal_points(fiducial_locations_df, quality_scores_df):
    
    median_scores = []
    for i in np.arange(0,4):
        median_score = quality_scores_df.iloc[:,i].median()
        median_scores.append(median_score)

    principal_points = []
    for i in range(len(quality_scores_df)):
        
        subpixel_fiducial_locations = fiducial_locations_df.iloc[i].values
        subpixel_quality_scores = quality_scores_df.iloc[i].values

        principal_point = hipp.core.compute_principal_point(subpixel_fiducial_locations, 
                                                            subpixel_quality_scores, 
                                                            median_scores)
        principal_points.append([principal_point])
        
    principal_points_df = pd.DataFrame(principal_points,columns=['principal_point'])
    
    return principal_points_df
    
def compute_mean_midside_corner_principal_point(df_corner, df_midside):
    df = pd.concat([pd.DataFrame(df_midside.principal_point.to_list(), columns=['midside_y',
                                                                               'midside_x']),
                    pd.DataFrame(df_corner.principal_point.to_list(), columns=['corner_y',
                                                                               'corner_x'])],axis=1)
    df['principal_point'] = list(zip(df[['midside_y', 'corner_y']].mean(axis=1),
                                     df[['midside_x', 'corner_x']].mean(axis=1)))
                                     
    return df
    
def compute_principal_point_from_proxies(df, verbose=True):
    distances = []
    principal_points = []
    intersection_angles = []
    verbose=False

    for index, row in df.iterrows():
        if verbose:
            print('Computing principal point for:', row['file_names'])
        p1 = (row['left_y'], row['left_x'])
        p2 = (row['right_y'], row['right_x'])
        principal_point_LR = hipp.math.midpoint(p1[1], p1[0], p2[1], p2[0])
        distances.append(hipp.math.distance(p1,p2))

        p1 = (row['top_y'], row['top_x'])
        p2 = (row['bottom_y'], row['bottom_x'])
        principal_point_TB = hipp.math.midpoint(p1[1], p1[0], p2[1], p2[0])
        distances.append(hipp.math.distance(p1,p2))
        
        # if no diametrically opposing proxies are found
        # use first viable combination of left/right x or top/bottom y
        # to estimate position
        if np.isnan(principal_point_LR).any() and np.isnan(principal_point_TB).any():
        
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row['left_y'], row['top_x'])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row['left_y'], row['bottom_x'])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row['right_y'], row['top_x'])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row['right_y'], row['bottom_x'])
        
            if np.isnan(principal_point_LR).any():
            
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row['left_y'], row['top_x'])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row['left_y'], row['bottom_x'])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row['right_y'], row['top_x'])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row['right_y'], row['bottom_x'])
                
        if np.isnan(principal_point_LR).any() and np.isnan(principal_point_TB).any():
#             if verbose:
            print('WARNING: Unable to estimate principal point for:', row['file_names'])
            print('WARNING: Using mean principal point estimate from image set instead.')
            principal_point = (np.nan,np.nan)
            principal_points.append(principal_point)
        else:
            principal_point = tuple(map(np.nanmean, zip(*(principal_point_TB, principal_point_LR))))
            principal_point = np.array([int(round(x)) for x in principal_point])
            principal_points.append(principal_point)
            if verbose:
                print('Principal point estimated at:', str(principal_point))
        
        proxy_locations    = np.array(list(zip(row.values[1::2], row.values[2::2])))
        intersection_angle = hipp.qc.compute_opposing_fiducial_intersection_angle(proxy_locations)
        intersection_angles.append(intersection_angle)
        if verbose:
            if not np.isnan(intersection_angle) and verbose:
                print('Intersection angle at principal point:', str(intersection_angle))
            elif verbose:
                print('Insufficient fiducial proxies (<4) detected to compute intersection angle.')
                
    # Use mean principal point estimate from image set to replace instance where < 2 proxies were found.
    df_tmp = pd.DataFrame(principal_points)
    principal_points = list(df_tmp.fillna(df_tmp.mean().round().astype(int)).astype(int).values)
        
    return principal_points, distances, intersection_angles

def validate_square_dim(image_files,
                        buffer_distance,
                        principal_points,
                        image_square_dim
                       ):
    new_square_dims = []

    for i,v in enumerate(image_files):
        ds = rasterio.open(v)
        h = ds.height + buffer_distance *2
        w = ds.width + buffer_distance *2
        pp_h = principal_points[i][0] + buffer_distance/2
        pp_w = principal_points[i][1] + buffer_distance/2

        tmp_h = (pp_h + image_square_dim/2)
        tmp_w = (pp_w + image_square_dim/2)

        if tmp_w > w:
            new_square_dims.append(image_square_dim - (tmp_w - w))
        if tmp_h > h:
            new_square_dims.append(image_square_dim - (tmp_h - h))

    if new_square_dims:
        new_square_dim = int(np.floor(np.nanmin(new_square_dims)))
        return new_square_dim
    else:
        return None
    
def create_fiducial_template(image_file,
                             df = None,
                             output_directory = 'fiducials',
                             output_file_name='fiducial.tif',
                             distance_around_fiducial=100):
     
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if isinstance(df,type(None)):
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
    
def create_midside_fiducial_proxies_template(image_file, 
                                             df = None,
                                             output_directory = 'input_data/fiducials',
                                             buffer_distance = 250,
                                             threshold= 50):
    
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    n, bins, patches = plt.hist(image_array.ravel()[::40],bins=256,range=(0,256))
#     plt.close()
    
#     p = find_peaks(n,prominence=1, width=1, height=n.max()/3)
#     p = find_peaks(n,prominence=100,width=1)
    
#     threshold = p[1]['right_bases'][0]
#     print(threshold)

    plt.vlines(threshold,0,n.max(),'r')
    image_array = hipp.image.threshold_and_add_noise(image_array, threshold=threshold)
    image_array = hipp.image.clahe_equalize_image(image_array)
    image_array = hipp.image.img_linear_stretch(image_array)
    
    image_array = hipp.core.pad_image(image_array,
                                      buffer_distance = buffer_distance)
    if isinstance(df,type(None)):
        print('Select inner most point to crop from for midside fiducial marker proxies,')
        print('in order from Left - Top - Right - Bottom.')
        df = hipp.tools.point_picker(image_file,
                                point_count = 4)
    
    df = df + buffer_distance
    
    left_fiducial   = (df.x[0],df.y[0])
    top_fiducial    = (df.x[1],df.y[1])
    right_fiducial  = (df.x[2],df.y[2])
    bottom_fiducial = (df.x[3],df.y[3])

    fiducials = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]
    
    dist_w, dist_h = buffer_distance, buffer_distance
    
    x_L = int(left_fiducial[0]-dist_w)
    x_R = int(left_fiducial[0])
    y_T = int(left_fiducial[1]-2*dist_w)
    y_B = int(left_fiducial[1]+2*dist_w)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'L.tif'),cropped)

    x_L = int(top_fiducial[0]-2*dist_h)
    x_R = int(top_fiducial[0]+2*dist_h)
    y_T = int(top_fiducial[1]-dist_h)
    y_B = int(top_fiducial[1])
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'T.tif'),cropped)

    x_L = int(right_fiducial[0])
    x_R = int(right_fiducial[0]+dist_w)
    y_T = int(right_fiducial[1]-2*dist_w)
    y_B = int(right_fiducial[1]+2*dist_w)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'R.tif'),cropped)

    x_L = int(bottom_fiducial[0]-2*dist_h)
    x_R = int(bottom_fiducial[0]+2*dist_h)
    y_T = int(bottom_fiducial[1])
    y_B = int(bottom_fiducial[1]+dist_h)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'B.tif'),cropped)
    
    return output_directory
    
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

def crop_image_from_file(image_file_principal_point_tuple,
                         image_square_dim,
                         output_directory = 'input_data/cropped_images',
                         buffer_distance = 250,
                         stretch_histogram = True,
                         clahe_enhancement = True):
    
    image_file, principal_point = image_file_principal_point_tuple
    
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image_array = hipp.core.pad_image(image_array,
                                      buffer_distance = buffer_distance)
    
    image_array = hipp.image.crop_about_point(image_array,
                                              principal_point,
                                              image_square_dim = image_square_dim)

    if clahe_enhancement:
        image_array = hipp.image.clahe_equalize_image(image_array)
    if stretch_histogram:
        image_array = hipp.image.img_linear_stretch(image_array)
    
    path, basename, extension = hipp.io.split_file(image_file)
    out = os.path.join(output_directory,basename+extension)
    cv2.imwrite(out,image_array)
    return out

def define_midside_windows(image_array,
                           reduce_left_window_by_fraction   = 0,
                           reduce_top_window_by_fraction    = 0,
                           reduce_right_window_by_fraction  = 0,
                           reduce_bottom_window_by_fraction = 0):
    
    half_image_height     = int(image_array.shape[0] / 2)
    quarter_image_height  = int(half_image_height / 2)

    half_image_width     = int(image_array.shape[1] / 2)
    quarter_image_width  = int(half_image_width / 2)
    
    midside_left    = [int(quarter_image_height + quarter_image_height*reduce_left_window_by_fraction),
                      int(half_image_height + quarter_image_height-quarter_image_height*reduce_left_window_by_fraction),
                      0, 
                      int(quarter_image_width - quarter_image_width*reduce_left_window_by_fraction)]

    midside_top     = [0,
                      int(quarter_image_height - quarter_image_height*reduce_top_window_by_fraction),
                      int(quarter_image_width + quarter_image_width*reduce_top_window_by_fraction),
                      int(half_image_width + quarter_image_width-quarter_image_width*reduce_top_window_by_fraction)]

    midside_right   = [int(quarter_image_height+ quarter_image_height*reduce_right_window_by_fraction),
                      int(half_image_height + quarter_image_height-quarter_image_height*reduce_right_window_by_fraction),
                      half_image_width + quarter_image_width + \
                           int(quarter_image_width*reduce_right_window_by_fraction),
                      image_array.shape[1]]

    midside_bottom  = [half_image_height + quarter_image_height + \
                           int(quarter_image_height*reduce_bottom_window_by_fraction),
                      image_array.shape[0],
                      int(quarter_image_width+ quarter_image_width*reduce_bottom_window_by_fraction),
                      int(half_image_width + quarter_image_width - + quarter_image_width*reduce_bottom_window_by_fraction)]
    
#     midside_left = [5900, 6500,0, 1500]
#     midside_top = [0, 500,6300, 7700]
#     midside_right = [5900, 6500, 12800, 13251]
#     midside_bottom = [11900, 12432,6400, 7700]
                     
    midside_windows = [midside_left, midside_top, midside_right, midside_bottom]
    
    return midside_windows

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

def define_center_window(image_array):
    
    # define window as slices: [y1:y2, x1:x2], so window = [y1, y2, x1, x2]
    
    half_image_height     = int(image_array.shape[0] / 2)
    quarter_image_height  = int(half_image_height / 2)

    half_image_width     = int(image_array.shape[1] / 2)
    quarter_image_width  = int(half_image_width / 2)
    
    center_window = [[int(np.round(half_image_height - quarter_image_height/2)),
                     int(np.round(half_image_height + quarter_image_height/2)),
                     int(np.round(half_image_width - quarter_image_width/2)),
                     int(np.round(half_image_width + quarter_image_width/2))]]
              
    
    return center_window

def detect_fiducials(slices,
                     template_array,
                     windows):

    matches = []
    quality_scores = []

    for index, slice_array in enumerate(slices):
        
        match_location, quality_score = hipp.core.match_template(slice_array,
                                                                 template_array)
                                         
        match = (windows[index][0] + match_location[0],
                 windows[index][2] + match_location[1])
        matches.append(match)
        quality_scores.append(quality_score)
        
    return matches, quality_scores

def detect_fiducial_proxies(image_file,
                            templates,
                            buffer_distance=250):

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
#     image_array = cv2.imread(image_file,cv2.IMREAD_COLOR)
#     image_array = image_array[:,:,0]
    
#     n, bins, patches = plt.hist(image_array.ravel()[::40],bins=256,range=(0,256))
#     plt.close()
#     p = find_peaks(n,prominence=10, width=1, height=n.max()/3)
#     threshold = p[1]['right_bases'][0]
#     image_array = hipp.image.threshold_and_add_noise(image_array, threshold=threshold)
    
    image_array = hipp.image.clahe_equalize_image(image_array)
    image_array = hipp.image.img_linear_stretch(image_array)
#     image_array = hipp.image.threshold_and_add_noise(image_array)
    
    image_array = hipp.core.pad_image(image_array,
                                      buffer_distance = buffer_distance)
    windows = hipp.core.define_midside_windows(image_array)
    slices = hipp.core.slice_image_frame(image_array,windows)
    
    matches = []
    quality_scores = []

    for index, slice_array in enumerate(slices):
        template = templates[index]
        
#         n, bins, patches = plt.hist(template.ravel()[::40],bins=256,range=(0,256))
#         plt.close()
#         p = find_peaks(n,prominence=10, width=1, height=n.max()/3)
#         threshold = p[1]['right_bases'][0]
#         template = hipp.image.threshold_and_add_noise(template.copy(), threshold=threshold)
    
#         template = hipp.image.clahe_equalize_image(template.copy())
#         template = hipp.image.img_linear_stretch(template.copy())
#         template = hipp.image.threshold_and_add_noise(template.copy())
        
        match_location, quality_score = hipp.core.match_template(slice_array,template)
        match = (windows[index][0] + match_location[0],
                 windows[index][2] + match_location[1])
        matches.append(match)
        quality_scores.append(quality_score)

    left, top, right, bottom = matches
    
    left_t_shape   = templates[0].shape
    top_t_shape    = templates[1].shape
    right_t_shape  = templates[2].shape
    bottom_t_shape = templates[3].shape

    left_fiducial   = (left[0]   + left_t_shape[0]/2 , left[1]   + left_t_shape[1])
    top_fiducial    = (top[0]    + top_t_shape[0]    , top[1]    + top_t_shape[1]/2)
    right_fiducial  = (right[0]  + right_t_shape[0]/2, right[1])
    bottom_fiducial = (bottom[0]                    , bottom[1] + bottom_t_shape[1]/2)

    matches = [left_fiducial,top_fiducial,right_fiducial,bottom_fiducial]
    
    return matches, quality_scores, image_file

def detect_high_res_fiducial(fiducial_crop_high_res_file,
                             template_high_res_zoomed_file,
                             distance_from_loc=200,
                             qc=True):
    
    fiducial_crop_high_res_array = cv2.imread(fiducial_crop_high_res_file,cv2.IMREAD_GRAYSCALE)
    template_high_res_zoomed_array = cv2.imread(template_high_res_zoomed_file,cv2.IMREAD_GRAYSCALE)
    
    match_location, quality_score = hipp.core.match_template(fiducial_crop_high_res_array,
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
    
    return match_location, quality_score
    

def detect_subpixel_fiducial_coordinates(image_file,
                                         image_array,
                                         matches,
                                         template_high_res_zoomed_file,
                                         labels = ['midside_left', 'midside_top', 'midside_right', 'midside_bottom'],
                                         distance_from_loc = 200,
                                         factor = 8,
                                         cleanup=True,
                                         qc=True):
    
    output_directory  ='tmp/fiducial_crop'
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    subpixel_fiducial_locations = []
    quality_scores = []

    for index, match_location in enumerate(matches):

        cropped_fiducial_file = hipp.core.crop_fiducial(image_file,
                                                        image_array,
                                                        match_location,
                                                        label=labels[index],
                                                        distance_from_loc = distance_from_loc,
                                                        output_directory='tmp/fiducial_crop')

        fiducial_crop_high_res_file = hipp.utils.enhance_geotif_resolution(cropped_fiducial_file,
                                                                           factor=factor)

        match_location_high_res, quality_score = hipp.core.detect_high_res_fiducial(fiducial_crop_high_res_file,
                                                                     template_high_res_zoomed_file,
                                                                     distance_from_loc=distance_from_loc,
                                                                     qc=qc)

        y,x = ((match_location_high_res[0]+int(distance_from_loc/2))/factor,
               (match_location_high_res[1]+int(distance_from_loc/2))/factor)

        subpixel_fiducial_location = y + match_location[0], x+match_location[1]

        subpixel_fiducial_locations.append(subpixel_fiducial_location)
        quality_scores.append(quality_score)
        
    if cleanup == True:
        shutil.rmtree('tmp/')
        
    return subpixel_fiducial_locations, quality_scores
    

def eval_and_compute_principal_point(P1, P1_score, P1_median_score,
                                     P2, P2_score, P2_median_score,
                                     threshold=0.01):
    """
    Evaluates fiducial point detection score and computes principal point estimates
    as midpoint between diametrically opposed fiducial markers.
    """
            
    if P1_median_score - P1_score < threshold and P2_median_score - P2_score < threshold:
        principal_point = hipp.math.midpoint(P1[1], P1[0], P2[1], P2[0])
        return principal_point
        
def eval_matches(df,
                 split_position_tuples=False,
                 threshold=0.01):
    """
    Replaces fiducial marker positions that received a low score with np.nan in place.
    A low score is determined by the difference between the median score for a given fiducial marker position
    across all images and a given score exceeding the threshold.
    Removes score columns and splits position tuples into seperate columns.
    """
    df = hipp.core.nan_low_scoring_fiducial_matches(df,threshold=threshold)
    
    columns = df.columns.values
    columns = [ x for x in columns if "score" not in x ]
    df = df[columns]
    
    if split_position_tuples:
        df = hipp.core.split_position_tuples(df)
    
    return df
    
def iter_crop_image_from_file(images,
                              principal_points,
                              image_square_dim,
                              output_directory = 'input_data/cropped_images',
                              buffer_distance = 250,
                              stretch_histogram = True,
                              clahe_enhancement = True,
                              verbose = True):
    
    print("Cropping images...")
    
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(images)) as pbar:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=True)-1)

        future = {pool.submit(hipp.core.crop_image_from_file,
                              img_pp,
                              image_square_dim,
                              buffer_distance=buffer_distance,
                              output_directory=output_directory,
                              stretch_histogram = stretch_histogram,
                              clahe_enhancement = clahe_enhancement): img_pp for img_pp in zip(images, principal_points)}
        results=[]
        for f in concurrent.futures.as_completed(future):
            r = f.result()
            pbar.update(1)
    print("Cropped images at:",output_directory)

def iter_detect_fiducial_proxies(images,
                                 templates,
                                 buffer_distance=250,
                                 verbose=False):
    print("Detecting fiducial proxies...")
    with tqdm(total=len(images)) as pbar:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=True)-1)
        future = {pool.submit(hipp.core.detect_fiducial_proxies,
                              image_file,
                              templates,
                              buffer_distance=buffer_distance): image_file for image_file in images}
        results=[]
        for f in concurrent.futures.as_completed(future):
            r = f.result()
            results.append(r)
            pbar.update(1)
    df = pd.DataFrame(results,columns=['match_locations',
                                       'scores',
                                       'file_names']).sort_values(by=['file_names']).reset_index(drop=True)
    return df
    
def load_midside_fiducial_proxy_templates(template_directory):
    
    # need to get rid of old jpg templates and recreate for all marker types
    l_tif_path = os.path.join(template_directory,'L.tif')
    l_jpg_path = os.path.join(template_directory,'L.jpg')
    
    assert (
        os.path.exists(l_tif_path) or os.path.exists(l_jpg_path)
    ), f"Fiducial marker files \"L.tif\" or \"L.jpg\" must exist in provided directory {template_directory}." 
    
    extension = '.tif' if os.path.exists(l_tif_path) else '.jpg'
    
    L = os.path.join(template_directory,'L' + extension)
    T = os.path.join(template_directory,'T' + extension)
    R = os.path.join(template_directory,'R' + extension)
    B = os.path.join(template_directory,'B' + extension)

        
    template_files = [L, T, R, B]
    templates = []

    for t in template_files:
        template = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
        templates.append(template)
    
    return templates

def match_template(image_array,
                   template_array):
    
#     image_array = np.where(image_array>200,image_array,0)
#     template_array = np.where(template_array>200,template_array,0)
    
    result = cv2.matchTemplate(image_array,template_array,cv2.TM_CCOEFF_NORMED)
    location = np.where(result==result.max())
    
    match_location = (location[0][0], location[1][0])
    quality_score = result.max()
    
    return match_location, quality_score
    
def merge_midside_df_corner_df(df_corner=None, 
                               df_midside=None,
                               file_name_column = 'fileName'):
    
    if isinstance(df_midside, Iterable) and isinstance(df_corner, Iterable):
        df = hipp.core.compute_mean_midside_corner_principal_point(df_corner, df_midside)
        
        del df_midside['principal_point']
        del df_corner['principal_point']
        
        df_detected = pd.merge(df_midside, df_corner, on=file_name_column)
        df_detected = pd.concat([df_detected,df['principal_point']],axis=1)
        df_detected = hipp.core.split_position_tuples(df_detected)
        return df_detected
        
    elif isinstance(df_midside, Iterable) and not isinstance(df_corner, Iterable):
        df_midside = hipp.core.split_position_tuples(df_midside)
        return df_midside
    
    elif isinstance(df_corner, Iterable) and not isinstance(df_midside, Iterable):
        df_corner = hipp.core.split_position_tuples(df_corner)
        return df_corner

def nan_low_scoring_fiducial_matches(df,threshold=0.01):
    """
    Replaces fiducial marker positions that received a low score with np.nan in place.
    A low score is determined by the difference between the median score for a given fidcuial marker position
    accross all images and a given score exceeding the threshold.
    """
    df = df.copy()
    for i in np.arange(1,5):
        fiducials = df.iloc[:,i].values
        corresponding_scores = df.iloc[:,i+4].values
        
        median_score = np.median(corresponding_scores)
        
        for index,value in enumerate(corresponding_scores):
            if median_score-value > threshold:
                fiducials[index]= np.nan
    return df

def nan_offset_fiducial_proxies(iter_detect_fiducial_proxies_df,
                                threshold_px = 50,
                                missing_proxy=None):
    
    df = pd.DataFrame(list(iter_detect_fiducial_proxies_df['match_locations'].values), 
                      columns=['left','top','right','bottom'])
    df.insert(0, 'file_names', iter_detect_fiducial_proxies_df['file_names'])
    df = hipp.core.split_position_tuples(df,skip=1)
    
    for key in df.keys()[1:]:
        offsets = df[key] - np.median(df[key])
        for index, value in enumerate(offsets):
            if abs(value) > threshold_px: # nan if offset from median position
                df.loc[df.index == index, key] = np.nan
    
    if missing_proxy   == 'left':
        df[['left_y'  ,   'left_x']]   = (np.nan,np.nan)
    elif missing_proxy == 'top':
        df[['top_y'   ,   'top_x']]    = (np.nan,np.nan)
    elif missing_proxy == 'right':
        df[['right_y' ,   'right_x']]  = (np.nan,np.nan)
    elif missing_proxy == 'bottom':
        df[['bottom_y',   'bottom_x']] = (np.nan,np.nan)
        
    return df
    
def pad_image(image_array,
              buffer_distance = 250):
    """
    Pad 2D np.array with zeros on all sides.
    """
    a=image_array.shape[0] + 2 * buffer_distance
    b=image_array.shape[1] + 2 * buffer_distance
    padded_img = np.zeros([a,b],dtype=np.uint8)
    padded_img[buffer_distance:buffer_distance+image_array.shape[0],
               buffer_distance:buffer_distance+image_array.shape[1]] = image_array
    return padded_img
    
    
def slice_image_frame(image_array, 
                      windows):

    slices = []
    for window in windows:
        slice_array = image_array[window[0]:window[1],
                                   window[2]:window[3]]
        slices.append(slice_array)
        
    return slices
    
def split_position_tuples(df,skip=1):
    df = df.copy()
    keys = df.keys().values[skip:]

    for key in keys:
        df_clean = pd.DataFrame(df[key].tolist(), 
                          index=df.index, 
                          columns=[key+'_y', key+'_x'])

        df = pd.concat([df,df_clean],axis=1)

    df = df.drop(keys, axis = 1)
    return df