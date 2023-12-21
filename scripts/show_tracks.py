from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from scripts.classes_fixed import *
from scripts.track_pairs import *

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def show_tracks(atl03path, atl08path, ax):
    
    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    
    vmin, vmax = np.inf, -np.inf
    
    for gt in tracks:
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            continue
        
        atl08 = ATL08(atl08path, gt)
        
        df = atl08.df.loc[:,['lat','lon', 'Eg']]
        
        vmin = min(df['Eg'].min(), vmin)
        vmax = max(df['Eg'].max(), vmax)
        
        sc = ax.scatter(df['lon'], df['lat'], c=df['Eg'], cmap = 'viridis', marker='o', label='Data Points', zorder=3, s=10)
        
    for gt in tracks:
        ax.scatter([], [], label=gt, c=[], vmin=vmin, vmax=vmax, cmap='viridis')
        
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, label='Eg Values')
    sc.set_clim(vmin, vmax)
        
    return ax
    
def map_setup(map_path, extent = None):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,7))
    if extent != None:
        ax.set_extent(extent)
    
    tif = rasterio.open(map_path)
    show(tif, ax=ax, transform=ccrs.PlateCarree())
    
    # Add labels, title, and legend
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Map of Tracks')
    # ax.legend()
    
    # Add latitude and longitude gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=1, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False  # Updated lines
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    return fig, ax