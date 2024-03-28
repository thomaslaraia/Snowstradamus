# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:42:34 2024

@author: s1803229
"""

from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, cmap2, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
import seaborn as sns
from scripts.classes_fixed import *
from scripts.track_pairs import *
from scripts.show_tracks import *
from scripts.parallel import pvpg_parallel

# Function to compute mean without the warning
def safe_nanmean(slice):
    if len(slice) == 0 or np.isnan(slice).all():
        return np.nan
    else:
        return np.nanmean(slice)
    
def FSC_dataframe(dirpath, csv_path):
    all_ATL03, all_ATL08 = track_pairs(dirpath)
    N = len(all_ATL03)
    
    df = pd.read_csv(csv_path).dropna().drop('Image', axis=1)
    
    pvpg = []
    mean_Eg_strong = []
    mean_Eg_weak = []
    mean_Ev_strong = []
    mean_Ev_weak = []
    msw_flags = []
    night_flags = []
    asrs = []
    
    for i in df['File Number']:
        coefs,means,msw_flag,night_flag,asr = pvpg_parallel(all_ATL03[int(i)], all_ATL08[int(i)], file_index = int(i),loss='arctan')
        pvpg.append(-coefs[0])
        mean_Eg_strong.append(safe_nanmean(means[0]))
        mean_Eg_weak.append(safe_nanmean(means[1]))
        mean_Ev_strong.append(safe_nanmean(means[2]))
        mean_Ev_weak.append(safe_nanmean(means[3]))
        msw_flags.append(msw_flag)
        night_flags.append(night_flag)
        asrs.append(asr)
    
    df['pvpg'] = pvpg
    df['mean_Eg_strong'] = mean_Eg_strong
    df['mean_Eg_weak'] = mean_Eg_weak
    df['mean_Ev_strong'] = mean_Ev_strong
    df['mean_Ev_weak'] = mean_Ev_weak
    df['msw_flag'] = msw_flags
    df['night_flag'] = night_flags
    df['asr'] = asrs
    
    #make categorical column
    df['Joint Snow'] = df['Joint Snow'].astype('category')
    df['FSC'] = df['FSC'].astype('category')
    df['Tree Snow'] = df['Tree Snow'].astype('category')
    
    df_pure = df.drop(['Camera','Date'], axis=1)
    
    return df, df_pure