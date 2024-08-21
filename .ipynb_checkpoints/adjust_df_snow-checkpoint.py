from scripts.FSC_dataframe_phoreal import *
import numpy as np
from pyhdf.SD import SD, SDC
import pyproj
from netCDF4 import Dataset

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save a concatenated dataframe from multiple directories.")
    parser.add_argument('output_pickle', type=str, help='Name of the output pickle file (without extension)')
    parser.add_argument('--input_pickle', type=str, default='five_sites_data', help='Name of input pickle file (without extension)')
    return parser.parse_args()

args = parse_args()

# Prepend '../' and append '.pkl' extension to the output filename
input_pickle_file = f"{args.input_pickle}.pkl"
df=pd.read_pickle(input_pickle_file)
# df = pd.read_pickle('five_sites_0-05_0-005box_snowreffed.pkl')

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
def hdf_to_sinusoidal(hdf_path):
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
    
    hdf.end()
    
    return xv, yv, data, ulx, uly, lrx, lry


df['MxD10A1F'] = None

# Process each date in the filtered DataFrame
for index, row in df.iterrows():
    # if index < 50:
    #     continue
    value = []
    div = 0
    for start in ['MOD','MYD']:
        place = row['camera']
    
        target_lat = row['latitude']
        target_lon = row['longitude']
        
        # Convert target coordinates to Sinusoidal projection using Transformer
        transformer = Transformer.from_crs("EPSG:4326", "+proj=sinu +R=6371007.181 +no_defs", always_xy=True)
        target_x, target_y = transformer.transform(target_lon, target_lat)
        
        hdf_dir = f'../data_store/data/{start}10A1F_{place}/'
        
        date_str = row['date']
        day_of_year = get_day_of_year(date_str)
        year = date_str[-4:]  # Extract year from 'dd/mm/yyyy'
        
        hdf_filename = find_matching_hdf(year, day_of_year, start, hdf_dir)
        
        if hdf_filename is None:
            print(f"HDF file for '{date_str}' not found.")
            continue
        
        hdf_path = os.path.join(hdf_dir, hdf_filename)
        
        # Convert HDF file to Sinusoidal coordinates and data array
        xv, yv, snow_cover_data, ulx, uly, lrx, lry = hdf_to_sinusoidal(hdf_path)
        
        # Compute distance in Sinusoidal projection to find nearest grid point
        distances = np.sqrt((xv - target_x)**2 + (yv - target_y)**2)
        nearest_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        
        # Extract value at the nearest grid cell
        value.append(snow_cover_data[nearest_idx])
        if snow_cover_data[nearest_idx] <= 100:
            div += 1

    if div == 1:
        v = np.min(value)
    else:
        v = np.mean(value)
    df.at[index,f'MxD10A1F'] = v
    # Print date and value
    # print(f"Date: {date_str}, Value at (-145.7514, 63.8811): {value}")
    # print(value)

# Prepend '../' and append '.pkl' extension to the output filename
output_pickle_file = f"{args.output_pickle}.pkl"

# Save dataframe to pickle file
df.to_pickle(output_pickle_file)
print(f'Dataframe saved to {output_pickle_file}')

# df.to_pickle('five_sites_data_snow')
