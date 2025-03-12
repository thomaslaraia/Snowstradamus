from scripts.imports import *
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer

def load_raster(filename):
    # Load the raster in EPSG:3067 projection
    with rasterio.open(filename) as src:
        # Print the CRS to verify it matches expectations
        # print(f"Raster CRS: {src.crs}")
        
        # Read the data
        data = src.read(1).astype(np.float32)
        
        # Get the affine transform (mapping coordinates to array indices)
        transform = src.transform

        crs = src.crs
        
        # Calculate the coordinates for the grid
        ny, nx = data.shape
        x_coords = np.linspace(transform[2], transform[2] + (nx * transform[0]), nx)
        y_coords = np.linspace(transform[5], transform[5] + (ny * transform[4]), ny)
        
        # Create an xarray DataArray with the correct coordinates
        data_array = xr.DataArray(data, coords=[y_coords, x_coords], dims=["y", "x"])
        
        # Filter out values greater than 100, assuming these are invalid
        data_array = data_array.where(data_array <= 100, drop=True)
        
    return data_array, crs

def average_pixel_value(data_array, center_x, center_y, buffer_size_m=500):
    # Define the bounding box in EPSG:3067 coordinates
    x_min = center_x - buffer_size_m / 2
    x_max = center_x + buffer_size_m / 2
    y_min = center_y - buffer_size_m / 2
    y_max = center_y + buffer_size_m / 2
    
    # Select the data within the bounding box
    sub_data = data_array.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    
    # Calculate the average of the valid data
    average_value = sub_data.nanmean().item()
    
    return average_value

def translate(latitude, longitude, crs):
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return x, y