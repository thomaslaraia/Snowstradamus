#from scripts.als import *

#filenames = ['sodankyla', 'delta', 'marcell', 'oregon']
#cameranames = ['sodankyla_full', 'delta_junction', 'marcell_MN', 'oregon_yp']

#df = pd.read_pickle('five_sites_data_snow.pkl')

#df['cc'] = None

#for i, f in enumerate(filenames):
#    filename = f'../data_store/data/{f}_cc.tiff'
    # filename = '../data/sodankyla_als/lasfiles/tiffs/list30.tiff'

#    data, crs = load_raster(filename)
#    print(crs)

#    for index, row in df.iterrows():
#        if row['camera'] == cameranames[i]:
#            x, y = translate(row['latitude'], row['longitude'], crs)
#            df.at[index,'cc'] = average_pixel_value(data, center_x = x, center_y = y, buffer_size_m=100)
#        else:
#            continue

#df.to_pickle('five_sites_data_snow_cc.pkl')

##################################################

from scripts.imports import *
df = pd.read_pickle('SCFG_accuracy_w_forest_frac.pkl')
# print(df.columns, df.Camera.unique(), df.x_coord.iloc[0])

df['cc'] = np.nan
# i =0
for index, row in df.iterrows():
    # i += 1
    if row['SCFG'] >= 0:
        if row['Camera'] == 'marcell':
            filename = '../data_store/data/marcell_cc.tiff'
        elif row['Camera'] == 'sodankyla':
            filename = '../data_store/data/sodankyla_cc.tiff'
        elif row['Camera'] == 'delta':
            filename = '../data_store/data/delta_cc.tiff'
        else:
            continue
        data, crs = load_raster(filename)
        x, y = translate(row['lat'], row['lon'], crs)
        df.at[index,'cc'] = average_pixel_value(data, center_x = x, center_y = y, buffer_size_m=500)
    # if i == 10000:
    #     break
    
# df.columns
df.to_pickle('work_in_progress.pkl')
