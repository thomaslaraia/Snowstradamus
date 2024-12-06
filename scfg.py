import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import os

df=pd.read_pickle('dataset.pkl')

# Function to convert date string from 'dd/mm/yyyy' to 'yyyymmdd'
def format_date_to_filename(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y%m%d')

# Function to read the NetCDF file and extract the value at given lat/lon
def get_scfg_value(date_str, lat, lon):
    # Convert the date to a format that matches the filename
    file_date = format_date_to_filename(date_str)
    
    # Extract year and month from the date string
    year = file_date[:4]
    month = file_date[4:6]
    
    # Construct the file path
    file_path = f'../data_store/data/SCFG/dap.ceda.ac.uk/{year}/{month}/{file_date}-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv2.0.nc'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return np.nan  # Return NaN if the file doesn't exist

    print(2)
    # Open the NetCDF file
    dataset = xr.open_dataset(file_path)
    
    # Extract the 'scfg' variable and clean the data (set values > 100 to NaN)
    scfg_data = dataset['scfg'].where(dataset['scfg'] <= 100)
    
    # Select the nearest value for the given latitude and longitude
    try:
        scfg_value = scfg_data.sel(lat=lat, lon=lon, method='nearest').isel(time=0).values
    except KeyError:
        scfg_value = np.nan  # If lat/lon doesn't exist in the file, return NaN

    scfg_data.close()
    
    return scfg_value
    
def compute_scfg_value(date_str, coord, radius_km=2, threshold=100, default_value=np.nan):
    """
    Compute the SCFG value as the average of values ≤ threshold within a radius_km,
    or set to default_value if all values exceed the threshold.

    Parameters:
        scfg_data (xarray.DataArray): The dataset with SCFG values and spatial dimensions (lat, lon).
        coord (tuple): The coordinate as (lon, lat) for which the value is computed.
        radius_km (float): The radius in kilometers to search around the given coordinate (default: 2).
        threshold (float): The maximum value to include in the average (default: 100).
        default_value (float): The value to assign if all values exceed the threshold (default: 200).

    Returns:
        float: The computed SCFG value.
    """
    file_date = formate_date_to_filename(date_str)
    
    year = file_date[:4]
    month = file_date[4:6]
    
    file_path = f'../data_store/data/SCFG/dap.ceda.ac.uk/{year}/{month}/{file_date}-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv2.0.nc'
    
    if not os.path.exists(file_path):
        return np.nan  # Return NaN if the file doesn't exist
        
    try:
        dataset = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        for i in range(len(coords)):
            for A in values:
                if j != 0:
                    A[i].append(A[i][-1])
                else:
                    A[i].append(np.nan)
        return np.nan
    scfg_data = dataset['scfg']
    
    # Unpack coordinates
    lon, lat = coord

    # Convert radius from kilometers to degrees
    radius_deg_lat = radius_km / 111.0  # 1 degree latitude ≈ 111 km
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(lat)))  # Adjust for latitude

    # Subset the data within the radius
    subset = scfg_data.sel(
        lat=slice(lat - radius_deg_lat, lat + radius_deg_lat),
        lon=slice(lon - radius_deg_lon, lon + radius_deg_lon)
    )

    # Mask values greater than the threshold
    valid_values = subset.where(subset <= threshold, drop=True)

    # Calculate the average or assign the default value
    if valid_values.size > 0:  # If there are valid values
        return valid_values.mean().item()
    else:  # If all values exceed the threshold
        return default_value

# Function to process the dataframe row by row
def process_dataframe(df):
    scfg_values = []
    
    for index, row in df.iterrows():
        date_str = row['date']  # Extract date in 'dd/mm/yyyy' format
        lat = row['latitude']   # Extract latitude
        lon = row['longitude']  # Extract longitude
        cam = row['camera']
        
        if cam == 'torgnon':
            r = 1
        else:
            r = 5
        
        # Filter out dates before October 2018 or after December 2020
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        if date_obj < datetime(2018, 10, 1) or date_obj > datetime(2021, 1, 1):
            scfg_values.append(np.nan)
        else:
            # Get the SCFG value for the current row
            scfg_value = compute_scfg_value(date_str, (lon, lat), radius_km = r)
            scfg_values.append(scfg_value)
    
    # Add the SCFG values as a new column in the dataframe
    df['SCFG'] = scfg_values
    
    return df

# Assuming df is the dataframe you want to process
# Process the dataframe
df = process_dataframe(df)

# Display the updated dataframe
# df[df['SCFG'] == np.nan]
df.topickle('dataset_SCFG.pkl')
