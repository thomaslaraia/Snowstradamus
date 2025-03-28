# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:42:34 2024

@author: s1803229
"""

from scripts.imports import *
import seaborn as sns
from scripts.classes_fixed import *
from scripts.track_pairs import *
from scripts.show_tracks import *
from scripts.parallel_blocks import pvpg_parallel

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
    
def FSC_dataframe(dirpath, csv_path, width=5, height=5, graph_detail = 0, threshold=2, small_box = 1, loss = 'arctan', alt_thresh=80, rebinned=False, method='normal'):
    all_ATL03, all_ATL08 = track_pairs(dirpath)
    N = len(all_ATL03)

    foldername = dirpath.split('/')[-2]
    
    excel_df = pd.read_csv(csv_path).drop('Image', axis=1)

    FSCs = []
    tree_snows = []
    joint_snows = []
    confidences = []
    
    dfs = []
    
    for i, (atl03_filepath, atl08_filepath) in enumerate(zip(all_ATL03, all_ATL08)):
        filedate = datetime_to_date(parse_filename_datetime(atl03_filepath))
        if ((excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername)).any():
            coords = (excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'x_coord'].iloc[0],\
                      excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'y_coord'].iloc[0])
            altitude = excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'Altitude'].iloc[0]
            DF = pvpg_parallel(dirpath, all_ATL03[int(i)], all_ATL08[int(i)],
                                                                coords = coords,width=width,height=height,
                                                                file_index = int(i),loss=loss, graph_detail=graph_detail,
                                                               altitude=altitude, threshold=threshold, small_box=small_box,\
                                                                  alt_thresh=alt_thresh, rebinned=rebinned, method=method)
            
            DF['FSC']=excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'FSC'].iloc[0]
            DF['TreeSnow']=excel_df.loc[(excel_df['Date']==filedate) & (excel_df['Camera']==foldername), 'Tree Snow'].iloc[0]
            
            dfs.append(DF)
    
    dfs_non_empty = [df for df in dfs if not df.empty]
    
    combined_df = pd.concat(dfs_non_empty, ignore_index=True)

    return combined_df
