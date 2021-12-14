import cv2
import holoviews as hv
import hvplot
import hvplot.xarray
import matplotlib
import numpy as np
import os
import panel as pn
import pathlib
import rasterio
import shutil
import xarray as xr
hv.extension('bokeh')

import hipp.image
import hipp.tools

import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)



"""
Library for interactive python tools.
"""

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
    
    
