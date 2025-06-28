import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

def parse_xml_metadata(xml_file):
    """Parse the XML metadata file to extract camera and illumination parameters"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1',
          'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    
    params = {
        'focal_length': float(root.find('.//isda:focal_length', ns).text),
        'pixel_resolution': float(root.find('.//isda:pixel_resolution', ns).text),
        'sun_azimuth': float(root.find('.//isda:sun_azimuth', ns).text),
        'sun_elevation': float(root.find('.//isda:sun_elevation', ns).text),
        'spacecraft_altitude': float(root.find('.//isda:spacecraft_altitude', ns).text)
    }
    
    return params

def load_image(image_path):
    """Load image in either PNG or other standard format"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img.astype(np.float32) / 255.0

def compute_surface_normals(dem, resolution):
    """Compute surface normals from DEM"""
    dzdx, dzdy = np.gradient(dem, resolution)
    magnitude = np.sqrt(dzdx**2 + dzdy**2 + 1)
    nx = -dzdx / magnitude
    ny = -dzdy / magnitude
    nz = 1 / magnitude
    return nx, ny, nz

def compute_illumination_angles(nx, ny, nz, sun_dir):
    """Compute illumination angles"""
    cos_incidence = nx * sun_dir[0] + ny * sun_dir[1] + nz * sun_dir[2]
    incidence = np.arccos(np.clip(cos_incidence, -1, 1))
    emission = np.arccos(np.clip(nz, -1, 1))
    return incidence, emission

def lunar_lambert(incidence, emission, k=0.6):
    """Lunar-Lambert photometric function"""
    mu = np.cos(incidence)
    mu0 = np.cos(emission)
    return (2 * k * mu0 / (mu0 + mu)) + (1 - k) * mu0

def shape_from_shading(img, metadata, iterations=100, learning_rate=0.01):
    """Estimate DEM from single image using shape-from-shading"""
    # Initialize DEM (starting with flat surface)
    dem = np.zeros_like(img)
    
    # Sun direction vector
    sun_az = np.radians(metadata['sun_azimuth'])
    sun_el = np.radians(metadata['sun_elevation'])
    sun_dir = np.array([
        np.cos(sun_el) * np.sin(sun_az),
        np.cos(sun_el) * np.cos(sun_az),
        np.sin(sun_el)
    ])
    
    # Pixel resolution (meters/pixel)
    resolution = metadata['pixel_resolution']
    
    # Initial albedo estimate
    albedo = np.mean(img)
    
    # Optimize using gradient descent
    for i in tqdm(range(iterations), desc="Shape from Shading"):
        # Compute surface normals
        nx, ny, nz = compute_surface_normals(dem, resolution)
        
        # Compute illumination angles
        incidence, emission = compute_illumination_angles(nx, ny, nz, sun_dir)
        
        # Compute modeled reflectance
        modeled = albedo * lunar_lambert(incidence, emission)
        
        # Compute gradient
        residual = img - modeled
        grad_x, grad_y = np.gradient(dem, resolution)
        
        # Update DEM
        dem += learning_rate * residual * (sun_dir[0] * grad_x + sun_dir[1] * grad_y + sun_dir[2])
        
        # Update albedo
        albedo = np.clip(np.mean(img / lunar_lambert(incidence, emission)), 0.05, 0.95)
    
    # Convert to elevation (meters)
    dem = metadata['spacecraft_altitude'] - (dem * metadata['pixel_resolution'])
    
    return gaussian_filter(dem, sigma=1)

def save_dem_image(dem, path, vmin=None, vmax=None):
    """Save DEM as a separate image file"""
    plt.figure(figsize=(8, 8))
    plt.imshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.axis('off')  # Turn off axis
    plt.colorbar(label='Elevation (meters)')
    plt.title('Digital Elevation Model')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved DEM image to {path}")

def visualize_results(img, dem, save_path=None):
    """Visualize input image and resulting DEM"""
    plt.figure(figsize=(12, 6))
    
    # Input image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.colorbar()
    
    # DEM
    plt.subplot(1, 2, 2)
    plt.imshow(dem, cmap='terrain')
    plt.title('Digital Elevation Model (meters)')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined visualization to {save_path}")
    plt.show()

def main():
    # File paths
    img_path = '/path/to/image'
    xml_file = '/path/to/xml/file
    output_dir = 'output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata = parse_xml_metadata(xml_file)
    
    # Load image
    print("Loading image...")
    img = load_image(img_path)
    
    # Generate DEM using shape-from-shading
    print("Estimating DEM from single image...")
    dem = shape_from_shading(img, metadata)
    
    # Save DEM as numpy array
    output_dem_npy = os.path.join(output_dir, 'lunar_dem.npy')
    np.save(output_dem_npy, dem)
    print(f"Saved DEM array to {output_dem_npy}")
    
    # Save DEM as separate image
    output_dem_img = os.path.join(output_dir, 'lunar_dem.png')
    save_dem_image(dem, output_dem_img)
    
    # Visualize and save combined results
    output_viz_path = os.path.join(output_dir, 'dem_result.png')
    visualize_results(img, dem, output_viz_path)

if __name__ == '__main__':
    main()
