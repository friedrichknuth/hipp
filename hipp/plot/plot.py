import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

import hipp

"""
Library for common plotting functions.
"""

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