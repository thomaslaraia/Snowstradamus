from scripts.imports import *
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.plot import show
from rasterio.windows import from_bounds as windows_from_bounds
from rasterio.transform import from_bounds
from scipy.spatial.distance import pdist
from pyproj import Transformer, CRS

def extract_nums(text, point='ul'):
    import re
    pattern = r'UpperLeftPointMtrs=\((.*?)\)' if point == 'ul' else r'LowerRightMtrs=\((.*?)\)'
    match = re.search(pattern, text)
    return tuple(map(float, match.group(1).split(','))) if match else (None, None)

def calculate_bounds(center_lat, center_lon, radius):
    radius_deg_lat = radius / 111000
    radius_deg_lon = radius / (111000 * np.cos(np.radians(center_lat)))

    lat_min = center_lat - radius_deg_lat
    lat_max = center_lat + radius_deg_lat
    lon_min = center_lon - radius_deg_lon
    lon_max = center_lon + radius_deg_lon

    return lat_min, lat_max, lon_min, lon_max

def get_day_of_year(date_str):
    date_obj = pd.to_datetime(date_str)
    return date_obj.strftime('%j')

def find_matching_hdf(year, day_of_year, start, hdf_dir):
    prefix = f'{start}10A1F.A{year}{day_of_year}'
    suffix = '.hdf'
    
    files = os.listdir(hdf_dir)
    for file in files:
        if file.startswith(prefix) and file.endswith(suffix):
            return file
    return None

def hdf_to_latlon(hdf_path):
    hdf = SD(hdf_path, SDC.READ)
    dataset = hdf.select('CGF_NDSI_Snow_Cover')
    data = dataset[:]
    
    ulx, uly = extract_nums(hdf.attributes()['StructMetadata.0'], point='ul')
    lrx, lry = extract_nums(hdf.attributes()['StructMetadata.0'], point='lr')
    nx, ny = data.shape
    
    xres = (lrx - ulx) / float(nx)
    yres = (uly - lry) / float(ny)
    
    x = np.linspace(ulx + xres / 2, lrx - xres / 2, nx)
    y = np.linspace(uly - yres / 2, lry + yres / 2, ny)
    xv, yv = np.meshgrid(x, y)
    
    sin_proj = pyproj.Proj(proj='sinu', R=6371007.181, no_defs=True)
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')

    transformer = Transformer.from_proj(sin_proj, wgs84_proj)
    longitude, latitude = transformer.transform(xv, yv)
    
    hdf.end()
    
    return latitude, longitude, data, ulx, uly, lrx, lry

def reproject_and_crop(snow_cover_data, ulx, uly, lrx, lry, lat_min, lat_max, lon_min, lon_max):
    src_transform = from_bounds(ulx, lry, lrx, uly, snow_cover_data.shape[1], snow_cover_data.shape[0])
    src_crs = rasterio.crs.CRS.from_proj4('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext')

    dst_transform, width, height = calculate_default_transform(
        src_crs, 'EPSG:4326', snow_cover_data.shape[1], snow_cover_data.shape[0], 
        left=ulx, bottom=lry, right=lrx, top=uly
    )

    reprojected_data = np.empty((height, width), np.float32)

    reproject(
        source=snow_cover_data,
        destination=reprojected_data,
        src_transform=src_transform,
        src_crs=src_crs,  # Sinusoidal projection
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        resampling=Resampling.nearest
    )

    reprojected_data[reprojected_data > 100] = np.nan

    # Recreate latitude and longitude arrays for the reprojected data
    reprojected_height, reprojected_width = reprojected_data.shape
    reprojected_lon, reprojected_lat = np.meshgrid(
        np.linspace(0, reprojected_width-1, reprojected_width),
        np.linspace(0, reprojected_height-1, reprojected_height)
    )
    reprojected_lon, reprojected_lat = dst_transform * (reprojected_lon, reprojected_lat)

    # Manually crop the reprojected data using the defined center and radius
    lat_mask = (reprojected_lat >= lat_min) & (reprojected_lat <= lat_max)
    lon_mask = (reprojected_lon >= lon_min) & (reprojected_lon <= lon_max)

    lat_indices = np.where(lat_mask.any(axis=1))[0]
    lon_indices = np.where(lon_mask.any(axis=0))[0]

    cropped_reprojected_data = reprojected_data[np.ix_(lat_indices, lon_indices)]

    return cropped_reprojected_data

def apply_elevation_mask(cropped_reprojected_data, elevation_data, tolerance, location_elevation, percentage=0.80):
    mask = np.zeros_like(cropped_reprojected_data, dtype=bool)
    pixel_size_y = elevation_data.shape[0] // cropped_reprojected_data.shape[0]
    pixel_size_x = elevation_data.shape[1] // cropped_reprojected_data.shape[1]

    for i in range(cropped_reprojected_data.shape[0]):
        for j in range(cropped_reprojected_data.shape[1]):
            elevation_block = elevation_data[i*pixel_size_y:(i+1)*pixel_size_y, j*pixel_size_x:(j+1)*pixel_size_x]
            elevation_values = elevation_block.flatten()
            elevation_values = elevation_values[~np.isnan(elevation_values)]
            
            within_tolerance = (elevation_values >= (location_elevation - tolerance)) & (elevation_values <= (location_elevation + tolerance))
            percentage_within_tolerance = np.sum(within_tolerance) / len(elevation_values)
            
            mask[i, j] = percentage_within_tolerance >= percentage

    masked_snow_cover_data = np.where(mask, cropped_reprojected_data, np.nan)
    return masked_snow_cover_data

def calculate_dissimilarity(data):
    valid_pixels = data[~np.isnan(data)]
    if len(valid_pixels) > 1:
        pairwise_distances = pdist(valid_pixels[:, None])  # Pairwise distances
        mean_dissimilarity = np.mean(pairwise_distances)  # Mean of pairwise distances
        return mean_dissimilarity
    return np.nan

def perform_analysis(start_date, end_date, center_lat, center_lon, radius, location_elevation, hdf_dir, tif_path, tolerances, percentage=0.80):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    lat_min, lat_max, lon_min, lon_max = calculate_bounds(center_lat, center_lon, radius)
    
    with rasterio.open(tif_path) as src:
        wgs84 = CRS.from_epsg(4326)
        tif_crs = src.crs
        transformer = Transformer.from_crs(wgs84, tif_crs, always_xy=True)
        center_x, center_y = transformer.transform(center_lon, center_lat)
    
        min_x = center_x - radius
        max_x = center_x + radius
        min_y = center_y - radius
        max_y = center_y + radius
    
        window = windows_from_bounds(min_x, min_y, max_x, max_y, src.transform)
        elevation_data = src.read(1, window=window)
    
    monthly_dissimilarity_results = {month: np.zeros(len(tolerances)) for month in range(1, 13)}
    monthly_unmasked_dissimilarity = {month: 0 for month in range(1, 13)}
    valid_days_count = {month: np.zeros(len(tolerances)) for month in range(1, 13)}

    for current_date in date_range:
        date_str = current_date.strftime('%Y-%m-%d')
        month = current_date.month
        day_of_year = get_day_of_year(date_str)
        year = date_str[:4]
        start = 'MOD'
        hdf_filename = find_matching_hdf(year, day_of_year, start, hdf_dir)

        if hdf_filename:
            hdf_path = os.path.join(hdf_dir, hdf_filename)
            latitude, longitude, snow_cover_data, ulx, uly, lrx, lry = hdf_to_latlon(hdf_path)

            cropped_reprojected_data = reproject_and_crop(snow_cover_data, ulx, uly, lrx, lry, lat_min, lat_max, lon_min, lon_max)

            for k, tolerance in enumerate(tolerances):
                masked_snow_cover_data = apply_elevation_mask(cropped_reprojected_data, elevation_data, tolerance, location_elevation,percentage=percentage)
                mean_dissimilarity = calculate_dissimilarity(masked_snow_cover_data)
                if not np.isnan(mean_dissimilarity):
                    monthly_dissimilarity_results[month][k] += mean_dissimilarity
                    valid_days_count[month][k] += 1

            mean_dissimilarity_unmasked = calculate_dissimilarity(cropped_reprojected_data)
            if not np.isnan(mean_dissimilarity_unmasked):
                monthly_unmasked_dissimilarity[month] += mean_dissimilarity_unmasked

    # Normalize results by valid days count
    for month in range(1, 13):
        monthly_dissimilarity_results[month] /= np.where(valid_days_count[month] == 0, 1, valid_days_count[month])
        monthly_unmasked_dissimilarity[month] /= np.max(valid_days_count[month])

    return monthly_dissimilarity_results, monthly_unmasked_dissimilarity