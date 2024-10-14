import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import os

df=pd.read_pickle('five_sites_data_snow_cc.pkl')

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

# Function to process the dataframe row by row
def process_dataframe(df):
    scfg_values = []
    
    for index, row in df.iterrows():
        date_str = row['date']  # Extract date in 'dd/mm/yyyy' format
        lat = row['latitude']   # Extract latitude
        lon = row['longitude']  # Extract longitude
        
        # Filter out dates before October 2018 or after December 2020
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        if date_obj < datetime(2018, 10, 1) or date_obj > datetime(2021, 1, 1):
            scfg_values.append(np.nan)
        else:
            # Get the SCFG value for the current row
            scfg_value = get_scfg_value(date_str, lat, lon)
            scfg_values.append(scfg_value)
    
    # Add the SCFG values as a new column in the dataframe
    df['SCFG'] = scfg_values
    
    return df

# Assuming df is the dataframe you want to process
# Process the dataframe
df = process_dataframe(df)

# Display the updated dataframe
# df[df['SCFG'] == np.nan]
df.topickle('SCFG.pkl')
