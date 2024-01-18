from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from scripts.classes_fixed import *
from scripts.track_pairs import *

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def show_tracks_only_atl03(atl03paths, ax, c = 'r', gtx = None):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03paths- Array of paths/to/atl03/file/
    ax - axis to plot figure on.
    gtx - array of strings to indicate which specific groundtracks you want to see
    """

    for atl03path in atl03paths:
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
        
    return ax

def show_tracks(atl03paths, atl08paths, ax, c = 'Eg', gtx = None, CBAR = 1):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03paths - Array of paths/to/atl03/file/
    atl08paths - Array of paths/to/atl08/file/
    ax - axis to plot figure on.
    c - value by which the tracks are coloured, either 'Eg' (default) or 'Ev'
    gtx - array of strings to indicate which specific groundtracks you want to see
    CBAR - set this to None if you don't want a colorbar. Useful if you are running this function for several files.
    """
    
    vmin, vmax = np.inf, -np.inf

    big_df = pd.DataFrame(columns=['lat', 'lon', c])
    
    for atl03path, atl08path in zip(atl03paths, atl08paths):
    
        # Set the tracks in order of Beam 1, 2, 3, 4, 5, 6
        A = h5py.File(atl03path, 'r')
        if list(A['orbit_info']['sc_orient'])[0] == 1:
        	strong = ['gt1r', 'gt2r', 'gt3r']
        	weak = ['gt1l', 'gt2l', 'gt3l']
        elif list(A['orbit_info']['sc_orient'])[0] == 0:
            strong = ['gt3l', 'gt2l', 'gt1l']
            weak = ['gt3r', 'gt2r', 'gt1r']
        else:
            print('Satellite in transition orientation.')
            A.close()
            return
        tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
        A.close()
        
        #override the previous if groundtracks are given as a parameter
        if gtx != None:
            tracks = gtx
        
        for gt in tracks:
            
            # Try to create an ATL03 structure
            try:
                atl03 = ATL03(atl03path, atl08path, gt)
            except (KeyError, ValueError, OSError) as e:
                continue
            
            # If that previous step works, then we should be able to link the ATL08 to the ATL03 without problems
            atl08 = ATL08(atl08path, gt)
            
            # Dataframe of the latitudes, longitudes, and Ev/Eg depending on parameter
            df = atl08.df.loc[:,['lat','lon', c]]
            big_df = pd.concat([big_df, df], ignore_index = True)
            
            # retrieve maximum and minimum Eg values
            vmin = min(df[c].min(), vmin)
            vmax = max(df[c].max(), vmax)
            
    # Plot each data point on the map created on map_setup(), coloured by its Eg/Ev value
    sc = ax.scatter(big_df['lon'], big_df['lat'], c=big_df[c], cmap = 'viridis', marker='o', label='Data Points', zorder=3, s=10)
        
    # Add colorbar
    sc.set_clim(vmin, vmax)
    if CBAR != None:
        cbar = plt.colorbar(sc, ax=ax, label=str(c) + ' Values')
        
    return ax
    
def map_setup(map_path, extent = None):
    """
    Sets up the plot to show the tracks on. Requires a geotiff file as basemap.

    map_path - path/to/map/
    extent - controls the map extent if want to focus on specific part of the map.d
    """
    
    # Create plot with relevant projection, and set extent if given as parameter
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,7))
    if extent != None:
        ax.set_extent(extent)
    
    # Open image and show on the plot
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
