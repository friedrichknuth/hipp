import os

import hipp

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
    
def enhance_geotif(geotif_file_name,
                   output_file_name=None,
                   outsize=None,
                   verbose=True,
                   print_call=False):
    
    if output_file_name is None:
        file_path, file_name, file_extension = hipp.io.split_file(geotif_file_name)
        output_file_name = os.path.join(file_path, 
                                        file_name+'_enhanced'+file_extension)
                                        
    call = ['gdal_translate',
            '-of','GTiff',
            '-co','TILED=YES',
            '-co','COMPRESS=LZW',
            '-co','BIGTIFF=IF_SAFER',
            '-outsize',str(outsize),str(outsize),
            '-r', 'cubic',
            geotif_file_name,
            output_file_name]
    
    if print_call==True:
        print(*call)
    
    else:
        hipp.io.run_command(call, verbose=verbose)
        return output_file_name






