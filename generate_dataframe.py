import argparse
import pandas as pd
import pickle
import os
from scripts.imports import *
import seaborn as sns
from scripts.classes_fixed import *
from scripts.track_pairs import *
from scripts.show_tracks import *
from scripts.parallel_blocks import *
import time
# from scripts.FSC_dataframe_phoreal import FSC_dataframe

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save a concatenated dataframe from multiple directories.")
    parser.add_argument('output_pickle', type=str, help='Name of the output pickle file (without extension)')
    parser.add_argument('--width', type=float, default=5, help='Width of the box (default: 0.05)')
    parser.add_argument('--height', type=float, default=5, help='Height of the box (default: 0.05)')
    parser.add_argument('--small_box', type=float, default=1, help='Size of the small box (default: 0.005)')
    parser.add_argument('--threshold', type=int, default=1, help='Data threshold value (default: 2)')
    parser.add_argument('--alt_thresh', type=int, default=80, help='Altitude threshold value (default: 90)')
    parser.add_argument('--rebinned', type=int, default=0, help='Rebinned into specified meter resolution')
    parser.add_argument('--method', type=str, default='bimodal', help='Method for probability distribution')
    parser.add_argument('--site', type=str, default='all', help='restrict to specific site if necessary')
    parser.add_argument('--outlier_removal', type=float, default=0.1, help='outlier_removal by elliptic envelope')
    parser.add_argument('--loss', type=str, default='linear', help='method for regression')
    return parser.parse_args()
    
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
    return datetime_obj.strftime('%Y-%m-%d')

# Main function
def main():
    args = parse_args()
    
    graph_detail = 0

    if args.outlier_removal == 0:
        args.outlier_removal = False

    if args.site == 'all':

        dirpaths = [
            '../data_store/data/sodankyla_full/',
            '../data_store/data/delta_junction/',
            '../data_store/data/marcell_MN/',
            '../data_store/data/lacclair/',
            '../data_store/data/torgnon/',
            '../data_store/data/oregon_yp/'
        ]

    else:

        dirpaths = [
            f'../data_store/data/{args.site}/'
        ]
        
    csv_path = 'snow_cam_details.xlsx'
    excel_df = pd.read_excel(csv_path).drop('Image', axis=1)
    
    #print(excel_df)
    
    output_pickle_file = f"{args.output_pickle}.pkl"
    checkpoint_file = f"{args.output_pickle}_checkpoint.pkl"
    
    # Load progress if checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            processed_indices, partial_df = pickle.load(f)
        print(f"Resuming from checkpoint. Already processed: {processed_indices}")
    else:
        processed_indices = set()
        partial_df = pd.DataFrame()

    for dir_idx, dirpath in enumerate(dirpaths):
        start_time = time.time()
        foldername = dirpath.split('/')[-2]
    
        all_ATL03, all_ATL08 = track_pairs(dirpath)
        total_indices = len(all_ATL03)

        for i in range(total_indices):
            # Skip already processed indices
            if (dir_idx, i) in processed_indices:
                continue

            try:
                filedate = datetime_to_date(parse_filename_datetime(all_ATL03[i]))
                #print(filedate)
                print
                
                if ((excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername)).any():
                    
                    coords = (excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'x_coord'].iloc[0],\
                              excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'y_coord'].iloc[0])
                    altitude = excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'Altitude'].iloc[0]
                    
                    df = pvpg_parallel(dirpath, all_ATL03[i], all_ATL08[i],
                                                                coords = coords,width=args.width,height=args.height,
                                                                file_index = i,loss=args.loss, graph_detail=graph_detail,
                                                               altitude=altitude, threshold=args.threshold, small_box=args.small_box,\
                                                                  alt_thresh=args.alt_thresh, rebinned=args.rebinned, method=args.method,
                                                                  outlier_removal=args.outlier_removal)
                                                                  
                    df['FSC'] = excel_df.loc[(excel_df['Date'] == filedate) & (excel_df['Camera'] == foldername), 'FSC'].iloc[0]
                    df['TreeSnow'] = excel_df.loc[(excel_df['Date']==filedate) & (excel_df['Camera']==foldername), 'Tree Snow'].iloc[0]
                    df['file_index'] = i
                    
                    
                
                #df = FSC_dataframe(
                #    dirpath, csvpath,
                #    width=args.width, height=args.height,
                #    graph_detail=0, threshold=args.threshold,
                #    small_box=args.small_box, alt_thresh=args.alt_thresh,
                #    rebinned=args.rebinned, method=args.method,
                #    outlier_removal=args.outlier_removal
                #)

                    # Append new data to the partial dataframe
                    if len(df) == 0:
                        print(dirpath, i)
                    else:
                        partial_df = pd.concat([partial_df, df], ignore_index=True)

                    # Update the processed indices
                    processed_indices.add((dir_idx, i))

                    # Save checkpoint after each index
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump((processed_indices, partial_df), f)


            except Exception as e:
                print(f"Error processing {dirpath} index {i}: {e}")
                continue

        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds (iteration {i})")

    # Save final results
    partial_df.reset_index(drop=True, inplace=True)
    partial_df.to_pickle(output_pickle_file)

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Dataframe saved to {output_pickle_file}")

if __name__ == '__main__':
    main()
