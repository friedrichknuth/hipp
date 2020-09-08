# Historical Image Pre-Processing
Library to pre-process scanned historical images for Structure from Motion (SfM) surface reconstruction and photogrammatric analysis. 

### Features

#### Data Query and Download
- Download imagery from historical image archives 
- Supported archives:
  - North American Glacier Photography (NAGAP)
  - USGS Earth Explorer (coming soon...)

#### Fiducial Marker Detection 
- Built-in application to create fiducial marker templates
- OpenCV template matching to detect fiducial marker coordinates
- Fiducial markers are detected at sub-pixel precision
- Can detect 4 midside and/or 4 corner fiducials
- Replaces poor matches with np.nan based on threshold
- Computes estimated principal point
- Quality Control
  - Principal point and fiducial marker coordinates are output as pickle file for inspection
  - Outputs window image around detected fiducial marker for visual verification
- Fiducial marker proxy detection (when actual fiducial markers are cropped out of image frame) (coming soon...)

#### Geometric Image Restitution 
- Computes affine transform between calibrated (true) fiducial marker coordinates and detected coordinates
  - Calibrated (true) fiducial marker coordinates are computed with respect to the detected principal point
- Affine transforms images for geometric restitution
  - Requires minimum of 3 successfully detected fiducial markers to perform restitution
- Crops transformed image about principal point to standard size
- Optional Linear or Contrast Limited Adaptive Historgram Equalization (CLAHE) to improve match point detection during SfM processing

### Examples
See [notebooks](./examples/) for processing examples.

### Installation
```
$ git clone https://github.com/friedrichknuth/hipp.git
$ cd ./hipp
$ conda env create -f environment.yml
$ conda activate hipp
$ pip install -e .
```

### References

Bradski, G. (2000). "The OpenCV Library". Dr. Dobb&#x27;s Journal of Software Tools.
