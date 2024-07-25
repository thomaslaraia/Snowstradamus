from scripts.als import *

filename = '../data_store/data/sodankyla_als/lasfiles/tiffs/list5.tiff'
# filename = '../data/sodankyla_als/lasfiles/tiffs/list30.tiff'

data = load_raster(filename)

# df=pd.read_pickle('five_sites_0-05_0-005box.pkl')
df = pd.read_pickle('five_sites_0-05_0-005box_n_photons_snowreffed.pkl')

df['cc'] = None

for index, row in df.iterrows():
    if row['camera'] == 'sodankyla_full':
        df.at[index,'cc'] = average_pixel_value(data, longitude=row['longitude'], latitude=row['latitude'], w=0.005)
    else:
        continue

df.to_pickle('five_sites_0-05_0-005box_n_photons_snowreffed.pkl')