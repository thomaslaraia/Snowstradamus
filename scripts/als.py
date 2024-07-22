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
        reprojected_data = np.empty((height, width), dtype=src.meta['dtype'])
        
        # Reproject the data
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        
        # Get the extent of the reprojected data
        extent = [transform * (0, 0), transform * (width, 0), transform * (0, height), transform * (width, height)]
        lon = np.linspace(extent[0][0], extent[1][0], width)
        lat = np.linspace(extent[2][1], extent[0][1], height)

        # Add buffer around the edges
        lon_buffered = np.linspace(extent[0][0] - 0.02, extent[1][0] + 0.02, width + 2)
        lat_buffered = np.linspace(extent[2][1] - 0.02, extent[0][1] + 0.02, height + 2)
        
        # Create a new data array with buffer
        buffered_data = np.full((height + 2, width + 2), np.nan, dtype=src.meta['dtype'])
        buffered_data[1:-1, 1:-1] = reprojected_data
        
        # Create an xarray DataArray
        data_array = xr.DataArray(buffered_data, coords=[lat_buffered, lon_buffered], dims=["lat", "lon"])
        
    return data_array

# def load_raster(filename):
#     # Load the raster
#     with rasterio.open(filename) as src:
#         # Define the source and destination CRS
#         src_crs = src.crs
#         dst_crs = 'EPSG:4326'
        
#         # Calculate the transform and the dimensions of the new image
#         transform, width, height = calculate_default_transform(
#             src_crs, dst_crs, src.width, src.height, *src.bounds)
        
#         # Define the metadata for the new image
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': dst_crs,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
        
#         # Create a new array to store the reprojected image
#         reprojected_data = np.empty((height, width), dtype=src.meta['dtype'])
        
#         # Reproject the data
#         reproject(
#             source=rasterio.band(src, 1),
#             destination=reprojected_data,
#             src_transform=src.transform,
#             src_crs=src.crs,
#             dst_transform=transform,
#             dst_crs=dst_crs,
#             resampling=Resampling.nearest)
        
#         # Get the extent of the reprojected data
#         extent = [transform * (0, 0), transform * (width, 0), transform * (0, height), transform * (width, height)]
#         lon = np.linspace(extent[0][0], extent[1][0], width)
#         lat = np.linspace(extent[2][1], extent[0][1], height)

#         # Create an xarray DataArray
#         data_array = xr.DataArray(reprojected_data, coords=[lat, lon], dims=["lat", "lon"])
        
#     return data_array

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
#     print(sub_data.shape)

    # Filter out values greater than 100
    valid_data = sub_data.where((sub_data >= 0) & (sub_data <= 100), drop=True)

    # Calculate the average of the valid data
    average_value = valid_data.mean().item()

    return average_value