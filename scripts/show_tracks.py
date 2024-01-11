from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from scripts.classes_fixed import *
from scripts.track_pairs import *

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def show_tracks_only_atl03(atl03path, ax, c = 'r', gtx = None):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03path - path/to/atl03/file/
    ax - axis to plot figure on.
    gtx - array of strings to indicate which specific groundtracks you want to see
    """
    
    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    if gtx != None:
        tracks = gtx
    
    for gt in tracks:
        
        try:
            atl03 = ATL03_without_ATL08(atl03path, gt)
        except (KeyError, ValueError, OSError) as e:
            continue
        
        df = atl03.df.loc[:,['lat','lon']]
        
        sc = ax.scatter(df['lon'], df['lat'], c=c, marker='o', zorder=3, s=4)
        
    #for gt in tracks:
        #ax.scatter([], [], label=gt, c=[], vmin=vmin, vmax=vmax, cmap='viridis')
        
    # Add colorbar
    #cbar = plt.colorbar(sc, ax=ax, label=str(c) + ' Values')
    #sc.set_clim(vmin, vmax)
        
    return ax

def show_tracks(atl03path, atl08path, ax, c = 'Eg', gtx = None, CBAR = 1):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03path - path/to/atl03/file/
    atl08path - path/to/atl08/file/
    ax - axis to plot figure on.
    c - value by which the tracks are coloured, either 'Eg' (default) or 'Ev'
    gtx - array of strings to indicate which specific groundtracks you want to see
    CBAR - set this to None if you don't want a colorbar. Useful if you are showing many, many tracks.
    """
    
    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    if gtx != None:
        tracks = gtx
    
    vmin, vmax = np.inf, -np.inf
    
    for gt in tracks:
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            continue
        
        
        atl08 = ATL08(atl08path, gt)
        
        df = atl08.df.loc[:,['lat','lon', c]]
        
        # retrieve maximum and minimum Eg values
        vmin = min(df[c].min(), vmin)
        vmax = max(df[c].max(), vmax)
        
        sc = ax.scatter(df['lon'], df['lat'], c=df[c], cmap = 'viridis', marker='o', label='Data Points', zorder=3, s=10)
        
    
        
    for gt in tracks:
        ax.scatter([], [], label=gt, c=[], vmin=vmin, vmax=vmax, cmap='viridis')
        
    # Add colorbar
    if CBAR != None:
        cbar = plt.colorbar(sc, ax=ax, label=str(c) + ' Values')
    sc.set_clim(vmin, vmax)
        
    return ax
    
def map_setup(map_path, extent = None):
    """
    Sets up the plot to show the tracks on. Requires a geotiff file as basemap.

    map_path - path/to/map/
    extent - controls the map extent if want to focus on specific part of the map.d
    """
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
