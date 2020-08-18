import cv2
import os
import pathlib
import numpy as np
import pandas as pd
from urllib.request import urlopen

import hipp


"""
Library query and download historical image data from archives.
"""

# NAGAP Functions
# TODO make a class for NAGAP specifics

def download_images_to_disk(camera_positions_file_name, 
                            output_directory='input_data/raw_images',
                            image_type='pid_tiff'):
                            
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(camera_positions_file_name)
    targets = dict(zip(df[image_type], df['fileName']))
                            
    for pid, file_name in targets.items():
        print('Downloading',file_name, image_type)
        img_gray = hipp.dataquery.download_image(pid)
        out = os.path.join(output_directory, file_name+'.tif')
        cv2.imwrite(out,img_gray)
        final_output = hipp.utils.optimize_geotif(out)
        os.remove(out)
        os.rename(final_output, out)
        
    return output_directory

def download_image(pid):
    base_url = 'https://arcticdata.io/metacat/d1/mn/v2/object/'
    url = base_url+pid
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    return image