from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from scripts.classes_fixed import *
from scripts.track_pairs import *
from shapely.geometry import Point, box as shapely_box

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
            
            sc = ax.scatter(df['lon'], df['lat'], c=c, marker='o', zorder=3, s=1)
        
    return ax

def make_box(coords, width=0.25, height=0.25):
    w = width
    h = height
    polygon = gpd.GeoDataFrame(geometry=[shapely_box(coords[0]-w/np.cos(np.radians(coords[1])), coords[1]-h, coords[0]+w/np.cos(np.radians(coords[1])), coords[1]+h)], crs="EPSG:4326")

    return polygon

def show_tracks(atl03paths, atl08paths, ax, coords, c = 'Eg', gtx = None, CBAR = None, w=.1, h=.1, landcover=None, vmax = 6):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03paths - Array of paths/to/atl03/file/
    atl08paths - Array of paths/to/atl08/file/
    ax - axis to plot figure on.
    c - value by which the tracks are coloured, either 'Eg' (default) or 'Ev'
    gtx - array of strings to indicate which specific groundtracks you want to see
    CBAR - set this to None if you don't want a colorbar. Useful if you are running this function for several files.
    """
    
    # vmax = -np.inf

    big_df = pd.DataFrame(columns=['lat', 'lon', c])

    polygon = make_box(coords, w,h)
    min_lon, min_lat, max_lon, max_lat = polygon.total_bounds
    
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

            atl08.df = atl08.df[(atl08.df['lon'] >= min_lon) & (atl08.df['lon'] <= max_lon) &\
                                (atl08.df['lat'] >= min_lat) & (atl08.df['lat'] <= max_lat)]
            if landcover != None:
                atl08.df = atl08.df[atl08.df['landcover'].isin([111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126])]
            
            # Dataframe of the latitudes, longitudes, and Ev/Eg depending on parameter
            df = atl08.df.loc[:,['lat','lon', c]]
            if big_df.shape[0] == 0:
                big_df = df
            else:
                big_df = pd.concat([big_df, df], ignore_index = True)

    
    # Plot each data point on the map created on map_setup(), coloured by its Eg/Ev value
    sc = ax.scatter(big_df['lon'], big_df['lat'], c=big_df[c], cmap = 'viridis', marker='o', label='Data Points', zorder=3, s=1)

    # vmax = big_df[c].max()
    # Add colorbar
    sc.set_clim(vmin = 0, vmax = vmax)
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
