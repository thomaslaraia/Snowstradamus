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

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    
    # Finding the index of the first appearance of 'ATL03_' or 'ATL08_'
    atl03_index = filename_only.find('ATL03_')
    atl08_index = filename_only.find('ATL08_')
    
    # Determining the split index based on which string appears first or if neither is found
    split_index = min(filter(lambda x: x >= 0, [atl03_index, atl08_index]))

    # Extracting yyyymmddhhmmss part
    date_str = filename_only[split_index + 6:split_index + 20]
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    
    return datetime_obj

def datetime_to_date(datetime_obj):
    return datetime_obj.strftime('%d/%m/%Y')
    
def FSC_dataframe(dirpath, csv_path, width=.05, height=.05, graph_detail = 0, threshold=10):
    all_ATL03, all_ATL08 = track_pairs(dirpath)
    N = len(all_ATL03)

    foldername = dirpath.split('/')[-2]
    
    excel_df = pd.read_csv(csv_path).drop('Image', axis=1).dropna()

    cameras = []
    dates = []
    pvpg = []
    mean_Eg_strong = []
    mean_Eg_weak = []
    mean_Ev_strong = []
    mean_Ev_weak = []
    msw_flags = []
    night_flags = []
    asrs = []
    FSCs = []
    tree_snows = []
    joint_snows = []
    confidences = []
    data_amount = []
    
    for i, (atl03_filepath, atl08_filepath) in enumerate(zip(all_ATL03, all_ATL08)):
        filedate = datetime_to_date(parse_filename_datetime(atl03_filepath))
        if ((excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername)).any():
            coords = (excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'x_coord'].iloc[0],\
                      excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'y_coord'].iloc[0])
            altitude = excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'Altitude'].iloc[0]
            coefs,means,msw_flag,night_flag,asr,DA = pvpg_parallel(all_ATL03[int(i)], all_ATL08[int(i)],
                                                                coords = coords,width=width,height=height,
                                                                file_index = int(i),loss='arctan', graph_detail=graph_detail,
                                                               altitude=altitude, threshold=threshold)
            if means != 0:
                cameras.append(foldername)
                dates.append(filedate)
                pvpg.append(-coefs[0])
                mean_Eg_strong.append(safe_nanmean(means[0]))
                mean_Eg_weak.append(safe_nanmean(means[1]))
                mean_Ev_strong.append(safe_nanmean(means[2]))
                mean_Ev_weak.append(safe_nanmean(means[3]))
                msw_flags.append(msw_flag)
                night_flags.append(night_flag)
                asrs.append(asr)
                FSCs.append(excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'FSC'].iloc[0])
                tree_snows.append(excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'Tree Snow'].iloc[0])
                joint_snows.append(FSCs[-1] + tree_snows[-1])
                confidences.append(excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'Certainty'].iloc[0])
                data_amount.append(DA)
    
    # Create an empty DataFrame
    df = pd.DataFrame()
    df['Location'] = cameras
    df['Date'] = dates
    df['pvpg'] = pvpg
    df['mean_Eg_strong'] = mean_Eg_strong
    df['mean_Eg_weak'] = mean_Eg_weak
    df['mean_Ev_strong'] = mean_Ev_strong
    df['mean_Ev_weak'] = mean_Ev_weak
    df['msw_flag'] = msw_flags
    df['night_flag'] = night_flags
    df['asr'] = asrs
    df['FSC'] = pd.Categorical(FSCs)
    df['Tree Snow'] = pd.Categorical(tree_snows)
    df['Joint Snow'] = pd.Categorical(joint_snows)
    df['Confidence'] = pd.Categorical(confidences)
    df['Data Amount'] = data_amount

    return df