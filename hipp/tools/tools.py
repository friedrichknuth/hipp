import cv2
import holoviews as hv
import hvplot
import hvplot.xarray
import numpy as np
import os
import panel as pn
import pathlib
import rasterio
import shutil
import xarray as xr
hv.extension('bokeh')

import hipp



"""
Library for python tools.
"""

def create_fiducial_template(image_file, 
                             output_directory = 'fiducials',
                             output_file_name='fiducial.tif',
                             cleanup=True,
                             distance_around_fiducial=100):
                     
    """
    Select center of fiducial marker
    """
    #TODO move to hipp.core
     
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    p = pathlib.Path('tmp/')
    p.mkdir(parents=True, exist_ok=True)

    temp_out = os.path.join('tmp/', 'temporary_image.tif')
    cv2.imwrite(temp_out,image_array)
    temp_out_optimized = hipp.utils.optimize_geotif(temp_out)
    os.remove(temp_out)
    os.rename(temp_out_optimized, temp_out)
    
    df = hipp.tools.point_picker(temp_out)

    fiducial = (df.x[0],df.y[0])
    
    x_L = int(fiducial[0]-distance_around_fiducial)
    x_R = int(fiducial[0]+distance_around_fiducial)
    y_T = int(fiducial[1]-distance_around_fiducial)
    y_B = int(fiducial[1]+distance_around_fiducial)
    cropped = image_array[y_T:y_B, x_L:x_R]
    
    out = os.path.join(output_directory,output_file_name)
    cv2.imwrite(out,cropped)
    
    if cleanup == True:
        shutil.rmtree('tmp/')
        
    
    return out
    
    
def point_picker(image_file_name,
                 point_count = 1):
    
    hv_image, subplot_width, subplot_height = hipp.tools.hv_plot_raster(image_file_name)

    points = hv.Points([])
    point_stream = hv.streams.PointDraw(source=points)

    app = (hv_image * points).opts(hv.opts.Points(width=subplot_width,
                                                  height=subplot_height,
                                                  size=5,
                                                  color='blue',
                                                  tools=["hover"]))

    panel = pn.panel(app)

    server = panel.show(threaded=True)

    condition = True
    while condition == True: 
        try:
            if len(point_stream.data['x']) == point_count:
                server.stop()
                condition = False
        except:
            pass

    df = point_stream.element.dframe()
    
    return df


def hv_plot_raster(image_file_name):
    
    src = rasterio.open(image_file_name)

    subplot_width  = hipp.tools.scale_down_number(src.shape[0])
    subplot_height = hipp.tools.scale_down_number(src.shape[1])

    da = xr.open_rasterio(src)
    
    da.values = hipp.image.img_linear_stretch(da.values)

    hv_image = da.sel(band=1).hvplot.image(rasterize=True,
                                      width=subplot_width,
                                      height=subplot_height,
                                      flip_yaxis=True,
                                      colorbar=False,
                                      cmap='gray')
                                      
    return hv_image, subplot_width, subplot_height
    
    
def scale_down_number(number, threshold=1000):
    while number > threshold:
        number = number / 2
    number = int(number)
    return number