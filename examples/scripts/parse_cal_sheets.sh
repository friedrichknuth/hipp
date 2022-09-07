#!/bin/bash

# Adapted from https://diging.atlassian.net/wiki/spaces/DCH/pages/5275668/Tutorial+Text+Extraction+and+OCR+with+Tesseract+and+ImageMagick

BPATH=$1  # Path to directory containing PDFs.
OPATH=$2  # Path to output directory.

# If the output path does not exist, create it.
if [ ! -d "$OPATH" ]; then
    mkdir -p "$OPATH"
fi

for FILEPATH in $BPATH*.pdf; do
  
    echo -n "Running OCR extraction..."
    OUTFILE=$OPATH$(basename $FILEPATH | cut -d. -f1)
    
    # Use imagemagick to convert the PDF to a high-rest multi-page TIFF.
    convert -density 300 "$FILEPATH" -depth 8 -strip -background white \
            -alpha off ./temp.tiff > /dev/null 2>&1
    
    # Then use Tesseract to perform OCR on the tiff. Discard afterwards.
    tesseract ./temp.tiff "$OUTFILE" > /dev/null 2>&1
    rm ./temp.tiff
    
    # Parse the OCR output and extract what we need.
    # This may change, depending on what we determine is useful.
    # Still requires manual grooming of the final output to catch 
    # errors that cannot be guarded against e.g. 2 being interpreted as 51
    
    # FOCAL LENGTH PARAMETERS
    echo 'focal_length' > "$OUTFILE"_extract.txt
    grep 'Calibrated Focal Length:' "$OUTFILE".txt | cut -d' ' -f 5 >> "$OUTFILE"_extract.txt
    
    echo ' ' >> "$OUTFILE"_extract.txt
    
    # RADIAL DISTORTION PARAMETERS
    echo 'radial_distortion' >> "$OUTFILE"_extract.txt
    grep -A6 'Degrees' "$OUTFILE".txt \
      | sed 's/Um/um/g' \
        | sed 's/o/0/g' \
          | sed 's/—/-/g' >> "$OUTFILE"_extract.txt
    
    echo ' ' >> "$OUTFILE"_extract.txt
    
    # TANGENTIAL DISTORTION
    echo 'tangential_distortion' >> "$OUTFILE"_extract.txt
    grep -A2 'Field Angle ' "$OUTFILE"_extract.txt | sed '/^$/d' >> "$OUTFILE"_extract.txt
    
    # RESOLVING POWER
    echo 'Field Angle: 0° 7.5° 15° 22.5° 30° 35° 45°' >> "$OUTFILE"_extract.txt
    grep -A2 'Field Angle:' "$OUTFILE".txt \
      | sed -n '/Field Angle/{n;p;n;p;}' >> "$OUTFILE"_extract.txt
    
    echo ' ' >> "$OUTFILE"_extract.txt
    
    # FIDUCIAL MARKS
    echo 'fiducial_marks' >> "$OUTFILE"_extract.txt
    grep -B9 'VIII. Distances Between Fiducial Marks' "$OUTFILE".txt \
      | sed '1d' | sed '$d' \
        | sed 's/[A-Za-z]//g' \
          | sed "s/[‘”“']//g" | sed 's/—/-/g' >> "$OUTFILE"_extract.txt
    
    echo ' ' >> "$OUTFILE"_extract.txt
    
    # DISTANCE BETWEEN FIDUCIAL MARKS
    echo 'fiducial_mark_distance' >> "$OUTFILE"_extract.txt
    grep -A2 'Corner fiducials (diagonals)' "$OUTFILE".txt | sed '$d' | sed "s/'/-/g" >> "$OUTFILE"_extract.txt
    grep -A2 'Midside fiducials' "$OUTFILE".txt | sed '$d' >> "$OUTFILE"_extract.txt
    grep -A5 'Corner fiducials (perimeter)' "$OUTFILE".txt \
      | sed 's/l/1/g' | sed 's/“ /-/g' | sed 's/[&*]/-/g'\
        | sed '/^$/d'  >> "$OUTFILE"_extract.txt
    
            
done