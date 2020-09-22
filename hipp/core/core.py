import cv2
from collections.abc import Iterable
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


def define_midside_windows(image_array):
    
    half_image_height     = int(image_array.shape[0] / 2)
    quarter_image_height  = int(half_image_height / 2)

    half_image_width     = int(image_array.shape[1] / 2)
    quarter_image_width  = int(half_image_width / 2)
    
    midside_left    = [quarter_image_height,
                      half_image_height + quarter_image_height,
                      0, 
                      quarter_image_width]

    midside_top     = [0,
                      quarter_image_height,
                      quarter_image_width,
                      half_image_width + quarter_image_width]

    midside_right   = [quarter_image_height,
                      half_image_height + quarter_image_height,
                      half_image_width + quarter_image_width,
                      image_array.shape[1]]


    midside_bottom  = [half_image_height + quarter_image_height,
                      image_array.shape[0],
                      quarter_image_width,
                      half_image_width + quarter_image_width]
                     
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

def match_template(image_array,
                   template_array):
    
    result = cv2.matchTemplate(image_array,template_array,cv2.TM_CCOEFF_NORMED)
    location = np.where(result==result.max())
    
    match_location = (location[0][0], location[1][0])
    quality_score = result.max()
    
    return match_location, quality_score
    
def merge_midside_df_corner_df(df_corner, 
                               df_midside,
                               image_file_name_column_name = 'fileName'):
    
    if isinstance(df_midside, Iterable) and isinstance(df_corner, Iterable):
        df = hipp.core.compute_mean_midside_corner_principal_point(df_corner, df_midside)
        
        del df_midside['principal_point']
        del df_corner['principal_point']
        
        df_detected = pd.merge(df_midside, df_corner, on=image_file_name_column_name)
        df_detected = pd.concat([df_detected,df['principal_point']],axis=1)
        df_detected = hipp.core.split_position_tuples(df_detected)
        return df_detected
        
    elif not isinstance(df_midside, Iterable) and isinstance(df_corner, Iterable):
        columns = [image_file_name_column_name, 
                   'midside_left_y', 
                   'midside_left_x', 
                   'midside_top_y',
                   'midside_top_x', 
                   'midside_right_y', 
                   'midside_right_x',
                   'midside_bottom_y', 
                   'midside_bottom_x']
        df_corner = hipp.core.split_position_tuples(df_corner)
        df_midside = pd.DataFrame(index=df_corner.index,columns=columns)
        df_midside[image_file_name_column_name] = df_corner[image_file_name_column_name]
        df_detected = pd.merge(df_midside, df_corner, on=image_file_name_column_name)
        return df_detected
    
    elif not isinstance(df_corner, Iterable) and isinstance(df_midside, Iterable):
        columns = [image_file_name_column_name,
                   'corner_top_left_y',
                   'corner_top_left_x', 
                   'corner_top_right_y', 
                   'corner_top_right_x',
                   'corner_bottom_right_y', 
                   'corner_bottom_right_x',
                   'corner_bottom_left_y', 
                   'corner_bottom_left_x']
        df_midside = hipp.core.split_position_tuples(df_midside)
        df_corner = pd.DataFrame(index=df_midside.index,columns=columns)
        df_corner[image_file_name_column_name] = df_midside[image_file_name_column_name]
        df_detected = pd.merge(df_midside, df_corner, on=image_file_name_column_name)
        return df_detected

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