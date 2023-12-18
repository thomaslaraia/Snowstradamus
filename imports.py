import matplotlib.pyplot as plt

## Imports
## System packages
import os, glob, pdb

## Data formatting and structure packages
import numpy as np
import h5py
import pandas as pd
import xarray as xr

## Geospatial packages
import geopandas as gpd
from pyproj import Proj, Transformer, CRS

## Plotting packages
from matplotlib import pyplot as plt
from matplotlib.cm import gist_earth as cmap

from scipy.odr import Model, Data, ODR

from datetime import datetime

import rasterio
from rasterio.plot import show

import cartopy.crs as ccrs
import cartopy.feature as cfeature
