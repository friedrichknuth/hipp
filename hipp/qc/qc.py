from collections.abc import Iterable
import numpy as np

import hipp

def compute_angle_diff(coordinates, coordinates_true):
    angle = hipp.qc.compute_opposing_fiducial_intersection_angle(coordinates)
    angle_true = hipp.qc.compute_opposing_fiducial_intersection_angle(coordinates_true)
    return abs(angle - angle_true)
    
def compute_coordinate_rmse(coordinates, coordinates_true):
    if np.isnan(coordinates).all() or np.isnan(coordinates_true).all():
        rmse = np.nan
    else:
        rmse = np.sqrt(np.nanmean((coordinates - coordinates_true)**2))
    return rmse

def compute_coordinate_distance_diff_rmse(midside_coordinates=None, 
                                          midside_coordinates_true=None,
                                          corner_coordinates=None,
                                          corner_coordinates_true=None):
    
    if isinstance(midside_coordinates, Iterable) and isinstance(corner_coordinates, Iterable):
        dist_midside      = hipp.qc.compute_opposing_fiducial_distances(midside_coordinates)
        dist_midside_true = hipp.qc.compute_opposing_fiducial_distances(midside_coordinates_true)
        dist_corner       = hipp.qc.compute_opposing_fiducial_distances(corner_coordinates)
        dist_corner_true  = hipp.qc.compute_opposing_fiducial_distances(corner_coordinates_true)
        dist = np.append(dist_midside, dist_corner)
        dist_true = np.append(dist_midside_true, dist_corner_true)
    elif isinstance(midside_coordinates, Iterable) and not isinstance(corner_coordinates, Iterable):
        dist      = hipp.qc.compute_opposing_fiducial_distances(midside_coordinates)
        dist_true = hipp.qc.compute_opposing_fiducial_distances(midside_coordinates_true)
        
    elif not isinstance(midside_coordinates, Iterable) and isinstance(corner_coordinates, Iterable):
        dist       = hipp.qc.compute_opposing_fiducial_distances(corner_coordinates)
        dist_true  = hipp.qc.compute_opposing_fiducial_distances(corner_coordinates_true)

    if np.isnan(dist).all() or np.isnan(dist_true).all():
        return np.nan
    else:
        rmse = np.sqrt(np.nanmean((dist - dist_true)**2))
    return rmse

def compute_opposing_fiducial_distances(coordinates):
    
    # Extract diametrically opposed fiducial marker coordinates. 
    # Order is the same for midside and corner fiducials.
    A0 = coordinates[0]
    A1 = coordinates[2]
    B0 = coordinates[1]
    B1 = coordinates[3]
    
    distA = hipp.math.distance(A0, A1)
    distB = hipp.math.distance(B0, B1)

    return np.array((distA, distB))

def compute_opposing_fiducial_intersection_angle(coordinates):
    
    # Extract diametrically opposed fiducial marker coordinates. 
    # Order is the same for midside and corner fiducials.
    A0 = coordinates[0]
    A1 = coordinates[2]
    B0 = coordinates[1]
    B1 = coordinates[3]
    
    m1 = hipp.math.slope(A0[0], A0[1], A1[0], A1[1])
    m2 = hipp.math.slope(B0[0], B0[1], B1[0], B1[1])
    
    intersection_angle = abs(hipp.math.intersection_angle(m1, m2))
    
    return intersection_angle


    
def convert_coordinates(coordinates, 
                        principal_point, 
                        scanning_resolution_mm = 0.02,
                        invert_y_axis = True):
    '''
    Converts pixel coordinates to camera reference system in mm.
    '''
    coordinates_mm            = coordinates.copy()
    principal_point_mm        = principal_point.copy()
    
    if invert_y_axis:
        coordinates_mm[:,1]       = coordinates_mm[:,1] * -1
        principal_point_mm[1]     = principal_point_mm[1] * -1
    
    # swap coordinate system origin to prinicpal point
    coordinates_mm = (coordinates_mm - principal_point_mm) * scanning_resolution_mm
    
    return coordinates_mm, principal_point_mm