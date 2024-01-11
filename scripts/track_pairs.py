import os

def track_pairs(dirpath, failed = False):
    """
    Goes through each file in a collection of ATL03 and ATL08 files from a given ROI, checks if a file is from ATL03, and if so searches for a corresponding ATL08 file. If it exists, they are appended to a list of filepaths. After collecting all existing file pairs, they are sorted by date.

    dirpath - path/to/data/
    failed - Default False, activate to receive an array of unpaired ATL03 files
    """
    all_ATL03 = []
    all_ATL08 = []
    
    failed_ATL03 = []

    # Extract characters after 'ATL03' and check for corresponding 'ATL08' file
    for file in os.listdir(dirpath):
        if 'ATL03' in file:
            # Extract string of characters after 'ATL03'
            suffix = file.split('ATL03')[1]

            # Check if corresponding 'ATL08' file exists
            atl08_file = os.path.join(dirpath, f'processed_ATL08{suffix}')
            if os.path.exists(atl08_file):
                all_ATL03.append(os.path.join(dirpath, file))
                all_ATL08.append(atl08_file)
            else:
                failed_ATL03.append(os.path.join(dirpath, file))

    all_ATL03.sort()
    all_ATL08.sort()
    failed_ATL03.sort()

    if failed == False:
        return all_ATL03, all_ATL08
    else:
        return all_ATL03, all_ATL08, failed_ATL03
