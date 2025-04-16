import tifffile
import holoviews as hv
import time
import panel as pn
import pandas as pd
import numpy as np
import xarray as xr
import hvplot.xarray



def point_picker(image_file_name: str, point_count = 1) -> pd.DataFrame:
    """
    Displays an interactive image viewer and allows the user to pick a number of points.

    Args:
        image_file_name (str): Path to the image file.
        point_count (int): Number of points the user is required to pick.

    Returns:
        list[tuple[int, int]]: List of (x, y) coordinates of selected points as integers.
    """
    # Generate image plot and get display dimensions
    hv_image, subplot_width, subplot_height = hv_plot_raster(image_file_name)

    # Initialize empty point layer and point drawing stream
    points = hv.Points([])
    point_stream = hv.streams.PointDraw(source=points)

    # Combine image and point layer into an interactive app
    app = (hv_image * points).opts(hv.opts.Points(
        width=subplot_width,
        height=subplot_height,
        size=5,
        color='blue',
        tools=["hover"]
    ))
    # Launch the interactive panel in a separate thread
    panel = pn.panel(app)
    server = panel.show(threaded=True)

    # Wait until the user has selected the desired number of points
    while True:
        if point_stream.data and len(point_stream.data.get('x', [])) == point_count:
            server.stop()
            break
        time.sleep(0.1)

    return point_stream.element.dframe()


def hv_plot_raster(image_file_name: str) -> tuple[hv.Overlay, int, int]:
    """
    Loads a TIFF image, converts it to grayscale, and prepares an hvPlot raster for visualization.

    Args:
        image_file_name (str): Path to the TIFF image file.

    Returns:
        tuple: (hvPlot object of the image, plot width, plot height)
    """
    # Read the TIFF image
    image = tifffile.imread(image_file_name)

    # Convert image to grayscale if it's multi-channel (e.g. RGB)
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=-1).astype(np.uint8)
    else:
        image_gray = image 

    # Convert to xarray DataArray with dimensions named "y" and "x"
    da = xr.DataArray(image_gray, dims=["y", "x"])

    # Adjust plot size to maintain correct aspect ratio
    plot_height, plot_width  = scale_down_shape(image_gray.shape)

    # Create an interactive raster image plot using hvPlot
    hv_plot = da.hvplot.image(cmap="gray", rasterize=True).opts(
        invert_yaxis=True,
        width=plot_width,
        height=plot_height,
        colorbar=False
    )
    return hv_plot , plot_width, plot_height


def scale_down_shape(shape: tuple[int, int], new_width: int = 800) -> tuple[int, int]:
    """
    Scales down the original image shape proportionally to a new width.

    Args:
        shape (tuple[int, int]): Original shape of the image (height, width).
        new_width (int): Desired width to scale down to (default is 800).

    Returns:
        tuple[int, int]: New shape (width, height) preserving aspect ratio.
    """
    # Calculate new height to preserve aspect ratio
    height = int(shape[1] / (shape[0]/new_width))
    return (new_width, height)
