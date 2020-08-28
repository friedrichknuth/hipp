import matplotlib.pyplot as plt
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