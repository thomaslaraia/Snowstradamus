from scripts.als import *

filenames = ['sodankyla', 'delta', 'marcell', 'oregon']
cameranames = ['sodankyla_full', 'delta_junction', 'marcell_MN', 'oregon_yp']

df = pd.read_pickle('five_sites_data_snow.pkl')

df['cc'] = None

for i, f in enumerate(filenames):
    filename = f'../data_store/data/{f}_cc.tiff'
    # filename = '../data/sodankyla_als/lasfiles/tiffs/list30.tiff'

    data, crs = load_raster(filename)
    print(crs)

    for index, row in df.iterrows():
        if row['camera'] == cameranames[i]:
            x, y = translate(row['latitude'], row['longitude'], crs)
            df.at[index,'cc'] = average_pixel_value(data, center_x = x, center_y = y, buffer_size_m=100)
        else:
            continue

df.to_pickle('five_sites_data_snow_cc.pkl')