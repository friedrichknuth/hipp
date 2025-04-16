import glob
import os
import cv2

import hipp.io
import hipp.utils
"""
Library for command line tools.
"""

def optimize_geotif(geotif_file_name,
                    output_file_name=None,
                    verbose=False,
                    print_call=False):
                   

    if output_file_name is None:
        file_path, file_name, file_extension = hipp.io.split_file(geotif_file_name)
        output_file_name = os.path.join(file_path, 
                                        file_name+'_optimized'+file_extension)
    
    call = ['gdal_translate',
            '-of','GTiff',
            '-co','TILED=YES',
            '-co','COMPRESS=LZW',
            '-co','BIGTIFF=IF_SAFER',
            geotif_file_name,
            output_file_name]
            
    if print_call==True:
        print(*call)
    
    else:
        hipp.io.run_command(call, verbose=verbose)
        return output_file_name

def optimize_geotifs(input_directory,
                     keep = False,
                     verbose=False):
    print('Optimizing tifs in', input_directory, 'with:')
    print(*['gdal_translate',
                '-of','GTiff',
                '-co','TILED=YES',
                '-co','COMPRESS=LZW',
                '-co','BIGTIFF=IF_SAFER'])
    tifs = sorted(glob.glob(os.path.join(input_directory,'*.tif')))
    output_tifs = []
    for tif in tifs:
        tif_optimized = hipp.utils.optimize_geotif(tif, verbose=verbose)
        if not keep:
            os.remove(tif)
            os.rename(tif_optimized, tif)
            output_tifs.append(tif)
        else:
            output_tifs.append(tif_optimized)
    return output_tifs
    
def enhance_geotif_resolution(geotif_file_name,
                              output_file_name=None,
                              factor=None,
                              verbose=False,
                              print_call=False):
    
    if output_file_name is None:
        file_path, file_name, file_extension = hipp.io.split_file(geotif_file_name)
        output_file_name = os.path.join(file_path, 
                                        file_name+'_high_res'+file_extension)
                                        
    img = cv2.imread(geotif_file_name,cv2.IMREAD_GRAYSCALE)
    w, h = img.shape[::-1]
    w, h = w*factor, h*factor
                                        
    call = ['gdal_translate',
            '-of','GTiff',
            '-co','TILED=YES',
            '-co','COMPRESS=LZW',
            '-co','BIGTIFF=IF_SAFER',
            '-outsize',str(w),str(h),
            '-r', 'cubic',
            geotif_file_name,
            output_file_name]
    
    if print_call==True:
        print(*call)
    
    else:
        hipp.io.run_command(call, verbose=verbose)
    return output_file_name






