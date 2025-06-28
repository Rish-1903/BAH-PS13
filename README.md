# BAH-PS13

# Lunar Digital Elevation Model (DEM) Generation using Shape-from-Shading

This project implements a Shape-from-Shading (SfS) algorithm to estimate lunar surface topography from single images captured by the Orbiter High Resolution Camera (OHRC). The method uses photometric cues and known illumination parameters to reconstruct surface elevation.

## Features

- XML metadata parsing for camera and illumination parameters
- Lunar-Lambert photometric model implementation
- Gradient-based optimization for surface reconstruction
- Visualization of input images and resulting DEMs
- DEM output in both numerical (NumPy) and visual formats

## Requirements

- Python 3.7+
- Required packages:
  - OpenCV (`cv2`)
  - NumPy
  - SciPy
  - Matplotlib
  - tqdm (for progress bars)
  - xml.etree.ElementTree (standard library)

Install requirements with:
```bash
pip install opencv-python numpy scipy matplotlib tqdm
```

## Usage

Place your input image and corresponding XML metadata file in the appropriate directories

Update the file paths in the main() function:

    img_path: Path to input PNG image

    xml_file: Path to XML metadata file

```bash
python lunar_sfs.py
```

Outputs

The script generates:

    lunar_dem.npy: NumPy array containing elevation values (meters)

    lunar_dem.png: Visual representation of the DEM

    dem_result.png: Side-by-side comparison of input image and DEM



