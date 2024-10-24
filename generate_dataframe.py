import argparse
import pandas as pd
from scripts.FSC_dataframe_phoreal import FSC_dataframe

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save a concatenated dataframe from multiple directories.")
    parser.add_argument('output_pickle', type=str, help='Name of the output pickle file (without extension)')
    parser.add_argument('--width', type=float, default=5, help='Width of the box (default: 0.05)')
    parser.add_argument('--height', type=float, default=5, help='Height of the box (default: 0.05)')
    parser.add_argument('--small_box', type=float, default=1, help='Size of the small box (default: 0.005)')
    parser.add_argument('--threshold', type=int, default=1, help='Data threshold value (default: 2)')
    parser.add_argument('--alt_thresh', type=int, default=80, help='Altitude threshold value (default: 90)')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    dirpaths = [
        '../data_store/data/sodankyla_full/',
        '../data_store/data/delta_junction/',
        '../data_store/data/marcell_MN/',
        '../data_store/data/lacclair/',
        '../data_store/data/torgnon/',
        '../data_store/data/oregon_yp/'
    ]
    csvpath = 'snow_cam_details.csv'

    for i, dirpath in enumerate(dirpaths):
        if i == 0:
            df = FSC_dataframe(dirpath, csvpath, width=args.width, height=args.height, graph_detail=0, threshold=args.threshold, small_box=args.small_box, alt_thresh=args.alt_thresh)
        else:
            df_ = FSC_dataframe(dirpath, csvpath, width=args.width, height=args.height, graph_detail=0, threshold=args.threshold, small_box=args.small_box, alt_thresh=args.alt_thresh)
            df = pd.concat([df, df_], axis=0)

    df.reset_index(drop=True, inplace=True)

    # Prepend '../' and append '.pkl' extension to the output filename
    output_pickle_file = f"{args.output_pickle}.pkl"

    # df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Save dataframe to pickle file
    df.to_pickle(output_pickle_file)
    print(f'Dataframe saved to {output_pickle_file}')

if __name__ == '__main__':
    main()
