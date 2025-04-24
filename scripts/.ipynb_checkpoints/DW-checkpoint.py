import pandas as pd
import xarray as xr
import rioxarray
from pathlib import Path

def find_dynamicworld_file(name, dw_dir='../scratch/data/DW'):
    dw_dir = Path(dw_dir)
    matches = list(dw_dir.glob(f'*{name}*.tif'))
    if not matches:
        raise FileNotFoundError(f"No DynamicWorld file found for name: {name}")
    return matches[0]

def assign_dw_labels(df, dw_dir='../scratch/data/DW'):
    unique_names = df['camera'].unique()
    df['DW'] = pd.NA  # Placeholder column

    for name in unique_names:
        filepath = find_dynamicworld_file(name, dw_dir)
        da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")
        mask = df['camera'] == name
        df.loc[mask, 'DW'] = da.sel(band=1).interp(
            y=("points", df.loc[mask, 'latitude'].values),
            x=("points", df.loc[mask, 'longitude'].values),
            method="nearest"
        ).values
    return df
