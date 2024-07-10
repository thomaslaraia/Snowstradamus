from scripts.FSC_dataframe_phoreal import *
from scripts.als import *
from pyhdf.SD import SD, SDC
import pyproj
from netCDF4 import Dataset

data = load_raster('../data_store/data/sodankyla_als/lasfiles/tiffs/list30.tiff')

# df=pd.read_pickle('five_sites_0-05_0-005box.pkl')
df = pd.read_pickle('five_sites_0-05_0-005box_snowreffed.pkl')

df = df[df['camera'] == 'sodankyla_full'

df['cc'] = None

small_box = 0.005

for index, row in df.iterrows():
    target_lat = row['latitude']
    target_lon = row['longitude']

    