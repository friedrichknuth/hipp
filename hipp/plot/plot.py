import concurrent
import cv2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pathlib
import psutil

import hipp.io
import hipp.plot

"""
Library for common plotting functions.
"""

def iter_plot_proxies(images,
                      proxy_locations_df,
                      principal_points,
                      buffer_distance = 250,
                      output_directory='qc/proxy_detection',
                      verbose=True):
    
    locations_no_buffer        = proxy_locations_df.iloc[:,1:] - buffer_distance
    locations_no_buffer        = locations_no_buffer.values.tolist()
    principal_points_no_buffer = np.array(principal_points) - buffer_distance

    pool = multiprocessing.Pool(processes=psutil.cpu_count(logical=False))
    for i in zip(images,locations_no_buffer,principal_points_no_buffer):
        pool.apply_async(hipp.plot.plot_proxies, args=(i,output_directory))
    pool.close()
    pool.join()

def plot_histogram(image_array,
                   figsize=(10, 5)):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n, bins, patches = ax.hist(image_array.ravel()[::40],
                        bins=256, 
                        range=(0,256))
    plt.show()

def plot_images(image_arrays,
                rows = 5,
                columns = 5,
                figsize=(10, 10),
                cmap = 'gray',
                title=None,
                labels=None,
                output_file_name=None):

    plt.figure(figsize=figsize)

    for i in range(rows*columns):
        ax = plt.subplot(rows, columns, i + 1)
        
        try:

            image = image_arrays[i]
            ax.imshow(image, cmap = cmap)
            ax.set_xticks(())
            ax.set_yticks(())
            
            if isinstance(labels, type(list())):
                ax.set_title(labels[i])
        except:
            ax.axis('off')
            pass
        
    if isinstance(title, type(str())):
        plt.suptitle(title, fontsize=15)
        plt.subplots_adjust(top=0.95)
        
    plt.tight_layout()
        
    if isinstance(output_file_name, type(str())):
        
        file_path, file_name, file_extension = hipp.io.split_file(output_file_name)
        
        p = pathlib.Path(file_path)
        p.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file_name)
        
def plot_restitution_qc(qc_df):
    
    output_directory = 'qc/restitution/'
    print('Image restitution qc plots in '+output_directory )
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    y_labels = ['mm', 'mm', 'degree', 'degree']
    
    titles  = ['Coordinates RMSE', 
               'Coordinates distance to Principal Point RMSE', 
               'Midside fiducial intersection angle at Principal Point difference',
               'Corner fiducial intersection angle at Principal Point difference']
               
    legend_labels = ['before transform', 'after transform',
                    'before transform', 'after transform',
                    'before transform', 'after transform',
                    'before transform', 'after transform']
                    
    output_names = ['coordinates_rmse',
                   'coordinates_pp_dist_rmse',
                   'midside_angle_diff',
                   'corner_angle_diff']
    
    for i in np.arange(1,5):
        fig,ax = plt.subplots(figsize=(12,5))
        key1 = qc_df.iloc[:,i].name
        key2 = qc_df.iloc[:,i+4].name
        qc_df[[key1,key2]].plot(ax=ax)
        ax.legend((legend_labels.pop(0),legend_labels.pop(0)))
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel(y_labels.pop(0))
        ax.set_title(titles.pop(0))

        out = os.path.join(output_directory,output_names.pop(0)+'.png')
        plt.savefig(out)
        # plt.close()

def plot_proxies(data,
                 output_directory=None):
    
    image_file        = data[0]
    proxies           = np.array(data[1])
    proxies_x         = proxies[1::2]
    proxies_y         = proxies[::2]
    principal_point   = data[2]
    principal_point_x = principal_point[1]
    principal_point_y = principal_point[0]
    
    
    if isinstance(output_directory, type(None)):
        output_directory='qc/proxy_detection'
        
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    path, name, ext = hipp.io.split_file(image_file)
    
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(image_array,cmap='gray')
    ax.scatter(proxies_x,proxies_y,color='lime',marker='.')
    ax.scatter(principal_point_x,principal_point_y,color='red', marker='.')
    plt.tight_layout()
    
    output_file_name = os.path.join(output_directory, name+'.png')
    
    fig.savefig(output_file_name)
    plt.close(fig)
    print("Fiducial proxy QC plot at:", output_file_name)
    return output_file_name
