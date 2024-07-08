from scripts.FSC_dataframe_phoreal import *

df=pd.read_pickle('five_sites_0-05_0-005box.pkl')

import re

def extract_nums(text, point = 'ul'):
    # Define the regular expression pattern

    if point == 'ul':
        pattern = r'UpperLeftPointMtrs=\((.*?)\)'
    else:
        pattern = r'LowerRightMtrs=\((.*?)\)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # Check if a match is found
    if match:
        # Extract the matched group (the characters between the parentheses)
        result = match.group(1)
        return parse_coordinates(result)
    else:
        return None, None

def parse_coordinates(coordinate_str):
    try:
        # Split the string by comma
        x_str, y_str = coordinate_str.split(',')
        
        # Convert the split strings to floats
        x = float(x_str)
        y = float(y_str)
        
        return x, y
    except ValueError:
        # Handle the case where conversion to float fails
        return None, None

# start = 'MYD'
# place='torgnon'
# target_lat = 45.8238
# target_lon = 7.5609

# hdf_dir = f'../data_store/data/{start}10A1F_{place}/'

# Function to get day of year (DOY) from a date in 'dd/mm/yyyy' format
def get_day_of_year(date_str):
    date_obj = pd.to_datetime(date_str, format='%d/%m/%Y')
    return date_obj.strftime('%j')

# Function to find HDF file that matches the pattern MOD10A1F.AYYYYDDD*.hdf
def find_matching_hdf(year, day_of_year, start, hdf_dirf):
    # Construct the prefix and suffix to match
    prefix = f'{start}10A1F.A{year}{day_of_year}'
    suffix = '.hdf'
    
    # List all files in the directory
    files = os.listdir(hdf_dir)
    
    # Find the matching HDF file
    for file in files:
        if file.startswith(prefix) and file.endswith(suffix):
            return file
    
    return None

# Function to convert HDF file to latitude and longitude arrays
def hdf_to_latlon(hdf_path):
    # Open HDF file and extract necessary data
    hdf = SD(hdf_path, SDC.READ)
    dataset = hdf.select('CGF_NDSI_Snow_Cover')
    data = dataset[:]
    
    # Get grid parameters from the metadata
    ulx, uly = extract_nums(hdf.attributes()['StructMetadata.0'], point='ul')
    lrx, lry = extract_nums(hdf.attributes()['StructMetadata.0'], point='lr')
    nx, ny = data.shape
    
    # Define the spatial resolution
    xres = (lrx - ulx) / float(nx)
    yres = (uly - lry) / float(ny)
    
    # Define the grid
    x = np.linspace(ulx + xres / 2, lrx - xres / 2, nx)
    y = np.linspace(uly - yres / 2, lry + yres / 2, ny)
    xv, yv = np.meshgrid(x, y)
    
    # Convert the projection to lat/lon using pyproj
    proj = pyproj.Proj(proj='sinu', R=6371007.181, no_defs=True)
    lon, lat = proj(xv, yv, inverse=True)
    
    # Close HDF file
    hdf.end()
    
    return lat, lon, data

df['MOD10A1F'] = None
df['MYD10A1F'] = None

# Process each date in the filtered DataFrame
for index, row in df.iterrows():
    for start in ['MOD','MYD']:
        place = row['camera']
        # print(place)
    
        target_lat = row['latitude']
        target_lon = row['longitude']
    
        # print(target_lat)
        # print(target_lon)
        
        hdf_dir = f'../data_store/data/{start}10A1F_{place}/'
        
        date_str = row['date']
        day_of_year = get_day_of_year(date_str)
        year = date_str[-4:]  # Extract year from 'dd/mm/yyyy'
        
        # Find the matching HDF file
        hdf_filename = find_matching_hdf(year, day_of_year, start, hdf_dir)
        
        if hdf_filename is None:
            print(f"HDF file for '{date_str}' not found.")
            continue
        
        hdf_path = os.path.join(hdf_dir, hdf_filename)
        
        # Convert HDF file to latitude, longitude, and data arrays
        latitude, longitude, snow_cover_data = hdf_to_latlon(hdf_path)
        
        # Find nearest indices for the given coordinates
        distances = np.sqrt((latitude - target_lat)**2 + (longitude - target_lon)**2)
        nearest_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        
        # Extract value at the nearest grid cell
        value = snow_cover_data[nearest_idx]

        df.at[index,f'{start}10A1F'] = value
    # Print date and value
    # print(f"Date: {date_str}, Value at (-145.7514, 63.8811): {value}")
    # print(value)

df.to_pickle('five_sites_0-05_0-005box_snowreffed.pkl')