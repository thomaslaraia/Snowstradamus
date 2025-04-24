"""
Simple function to pull in all the usual imports, saves some thinking time.
"""

import matplotlib.pyplot as plt
import re

## Imports
## System packages
import os, glob, pdb

## Data formatting and structure packages
import numpy as np
import h5py
import pandas as pd
import xarray as xr
import seaborn as sns

## Geospatial packages
import geopandas as gpd
from pyproj import Proj, Transformer, CRS, transform

## Plotting packages
from matplotlib import pyplot as plt
from matplotlib.cm import gist_earth as cmap
from matplotlib.cm import Dark2 as cmap2
from matplotlib.cm import tab20 as cmap3

from scipy.odr import Model, Data, ODR
from scipy.stats import mode

from datetime import datetime

import rasterio
from rasterio.plot import show
import rioxarray

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image

from pyhdf.SD import SD, SDC
import pyproj
from netCDF4 import Dataset

from pathlib import Path