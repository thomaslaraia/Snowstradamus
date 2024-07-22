from scripts.imports import *
from rasterio.warp import calculate_default_transform, reproject, Resampling

def load_raster(filename):
    # Load the raster
    with rasterio.open(filename) as src:
        # Define the source and destination CRS
        src_crs = src.crs
        dst_crs = 'EPSG:4326'
        
        # Calculate the transform and the dimensions of the new image
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Define the metadata for the new image
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create a new array to store the reprojected image
        reprojected_data = np.empty((height, width), dtype=np.float32) #dtype=np.float32 or src.meta['dtype']
        
        # Reproject the data
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        
        # # Replace zeros with np.nan to handle missing data
        # reprojected_data[reprojected_data == 0] = np.nan
        
        # Get the extent of the reprojected data
        extent = [transform * (0, 0), transform * (width, 0), transform * (0, height), transform * (width, height)]
        lon = np.linspace(extent[0][0], extent[1][0], width)
        lat = np.linspace(extent[2][1], extent[0][1], height)
        
        # Create an xarray DataArray
        data_array = xr.DataArray(reprojected_data, coords=[lat, lon], dims=["lat", "lon"])
        
    return data_array

def add_buffer(data_array, latitude, buffer_size_m=500, increment_m=30, buffer_value=250):
    # Constants for conversion
    degrees_per_meter_lat = 1 / 111320
    degrees_per_meter_lon = degrees_per_meter_lat / np.cos(np.radians(latitude))
    
    # Calculate the buffer size and increment in degrees
    buffer_size_lat = buffer_size_m * degrees_per_meter_lat
    buffer_size_lon = buffer_size_m * degrees_per_meter_lon
    
    increment_lat = increment_m * degrees_per_meter_lat
    increment_lon = increment_m * degrees_per_meter_lon
    
    # Get the current coordinates
    lon = data_array.lon.values
    lat = data_array.lat.values
    height, width = data_array.shape
    
    # Calculate the padding width in terms of number of increments
    pad_width_lat = int(buffer_size_m / increment_m)
    pad_width_lon = int(buffer_size_m / increment_m)
    
    # Create padded longitude and latitude arrays
    lon_buffered = np.pad(
        lon, 
        pad_width=pad_width_lon, 
        mode='linear_ramp', 
        end_values=(lon[0] - buffer_size_lon, lon[-1] + buffer_size_lon)
    )
    lat_buffered = np.pad(
        lat, 
        pad_width=pad_width_lat, 
        mode='linear_ramp', 
        end_values=(lat[0] - buffer_size_lat, lat[-1] + buffer_size_lat)
    )
    
    # Create a new data array with buffer
    buffered_data = np.full(
        (height + 2 * pad_width_lat, width + 2 * pad_width_lon), 
        buffer_value, 
        dtype=data_array.dtype
    )
    buffered_data[pad_width_lat:-pad_width_lat, pad_width_lon:-pad_width_lon] = data_array.values
    
    # Create an xarray DataArray
    buffered_data_array = xr.DataArray(buffered_data, coords=[lat_buffered, lon_buffered], dims=["lat", "lon"])
    
    return buffered_data_array

def average_pixel_value(data_array, longitude, latitude, w):
    # Calculate the bounding box
    lat_min = latitude - w / 2
    lat_max = latitude + w / 2
    lon_min = longitude - w / (2 * np.cos(np.radians(latitude)))
    lon_max = longitude + w / (2 * np.cos(np.radians(latitude)))

    # data.plot(cmap='gray')
    # Select the data within the bounding box
    sub_data = data_array.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    # sub_data.plot(cmap='gray')

    # Filter out values greater than 100
    valid_data = sub_data.where((sub_data >= 0) & (sub_data <= 100), drop=True)
    # valid_data.plot(cmap='gray')

    # Calculate the average of the valid data
    average_value = valid_data.mean().item()

    return average_value