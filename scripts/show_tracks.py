from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from scripts.classes_fixed import *
from scripts.track_pairs import *
from scripts.DW import *
from shapely.geometry import Point, box as shapely_box

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import sys

sys.path.insert(1,'/home/s1803229/src/PhoREAL')
# sys.path.insert(1,'C:/Users/s1803229/Documents/PhoREAL')

from phoreal.reader import get_atl03_struct, get_atl08_struct
from phoreal.binner import rebin_atl08

import folium
import matplotlib.colors as colors
import matplotlib.cm as cm

def show_colorbar(cmap='viridis', vmin=0, vmax=6, label='Eg'):
    fig, ax = plt.subplots(figsize=(6, 0.5))
    fig.subplots_adjust(bottom=0.5)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=plt.colormaps[cmap], norm=norm)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(label, fontsize=10)
    plt.show()

def plot_on_folium(df, c='Eg', cmap='viridis', zoom_start=12, background = 'OpenStreetMap'):
    # Drop any invalid data just in case
    df = df.dropna(subset=['latitude', 'longitude', c])
    df = df[np.isfinite(df[c])]

    # Set vmin and vmax
    vmin = 0
    vmax = df[c].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.colormaps[cmap]

    # Center map on average coordinates
    lat_center = df['latitude'].mean()
    lon_center = df['longitude'].mean()
    m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_start, tiles='OpenStreetMap')

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Plot each point
    for _, row in df.iterrows():
        color = colormap(norm(row[c]))
        hex_color = colors.to_hex(color)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color=hex_color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    return m

import contextily as ctx
from shapely.geometry import Point, box
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def plot_static_map_with_box(df, coords, c='Eg', cmap='viridis', vmin=0, vmax=6, save='no', w=4, h=4):
    lon_center, lat_center = coords
    km_to_deg_lat = 1 / 111  # degrees per km in latitude
    km_to_deg_lon = 1 / (111 * np.cos(np.radians(lat_center)))  # degrees per km in longitude

    # Compute the box size in degrees
    half_box_deg_lat = h * km_to_deg_lat
    half_box_deg_lon = w * km_to_deg_lon
    pad_deg_lat = h*0.2 * km_to_deg_lat
    pad_deg_lon = w*0.2 * km_to_deg_lon

    # Define the bounding box and padded extent
    box_geom = box(lon_center - half_box_deg_lon,
                   lat_center - half_box_deg_lat,
                   lon_center + half_box_deg_lon,
                   lat_center + half_box_deg_lat)

    extent = [lon_center - half_box_deg_lon - pad_deg_lon,
              lon_center + half_box_deg_lon + pad_deg_lon,
              lat_center - half_box_deg_lat - pad_deg_lat,
              lat_center + half_box_deg_lat + pad_deg_lat]

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")

    # Convert to Web Mercator for basemap
    gdf_web = gdf.to_crs(epsg=3857)
    box_web = gpd.GeoSeries([box_geom], crs="EPSG:4326").to_crs(epsg=3857)
    extent_web = gpd.GeoSeries([Point(x, y) for x, y in [
        (extent[0], extent[2]), (extent[1], extent[3])
    ]], crs="EPSG:4326").to_crs(epsg=3857).total_bounds

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_web.plot(ax=ax, column=c, cmap=cmap, markersize=3, legend=True, vmin=vmin, vmax=vmax)
    box_web.boundary.plot(ax=ax, color='red', linewidth=1.5, linestyle='--')

    ax.set_xlim(extent_web[0], extent_web[2])
    ax.set_ylim(extent_web[1], extent_web[3])
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    # Add axis gridlines and lat/lon labels
    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)
    from pyproj import Transformer

    # Transformer to go from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    # Format axis ticks to show degrees
    def format_lon(x, _): 
        lon, _ = transformer.transform(x, 0)
        return f"{lon:.2f}°"
    
    def format_lat(y, _): 
        _, lat = transformer.transform(0, y)
        return f"{lat:.2f}°"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ax.set_axis_off()
    plt.title(f"{c} values with 8×8 km box around ({lat_center:.4f}, {lon_center:.4f})")
    plt.tight_layout()
    if save != 'no':
        plt.savefig(f'{save}.jpg')
    plt.show()

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

def make_box(coords, width_km=4, height_km=4):
    # Convert width and height from kilometers to degrees
    km_per_degree_lat = 111  # Kilometers per degree of latitude
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))  # Kilometers per degree of longitude at given latitude

    # Convert the input width and height from kilometers to degrees
    width_deg = width_km / km_per_degree_lon
    height_deg = height_km / km_per_degree_lat

    # Create the bounding box using converted degrees
    polygon = gpd.GeoDataFrame(
        geometry=[
            shapely_box(
                coords[0] - width_deg, coords[1] - height_deg, 
                coords[0] + width_deg, coords[1] + height_deg
            )
        ], 
        crs="EPSG:4326"
    )
    
    return polygon

def show_tracks(atl03paths, atl08paths, coords, altitude, c = 'Eg', gtx = None, CBAR = None, w=4, h=4, landcover=None,
               res_field = 'alongtrack', rebinned=0, sat_flag = False, DW=0):
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

    big_df = pd.DataFrame(columns=['latitude', 'longitude', c])

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
            sub = []
            for i in gtx:
                sub.append(tracks[i-1])
            tracks = sub
        
        for gt in tracks:
            
            # Try to create an ATL03 structure
            try:
                atl03 = get_atl03_struct(atl03path, gt, atl08path)
                atl08 = get_atl08_struct(atl08path, gt, atl03)
            except (KeyError, ValueError, OSError) as e:
                continue

            atl03.df = atl03.df[(atl03.df['lon_ph'] >= min_lon) & (atl03.df['lon_ph'] <= max_lon) &\
                                    (atl03.df['lat_ph'] >= min_lat) & (atl03.df['lat_ph'] <= max_lat)]
            
            atl08.df = atl08.df[(atl08.df['longitude'] >= min_lon) & (atl08.df['longitude'] <= max_lon) &\
                                    (atl08.df['latitude'] >= min_lat) & (atl08.df['latitude'] <= max_lat)]

            # print(1)
            if rebinned > 0:
                try:
                    atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)
                except (KeyError, ValueError, OSError) as e:
                    continue
            # print(2)

            if DW != 0:
                foldername = DW.split('/')[-2]
                filepath = find_dynamicworld_file(foldername)
                da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")
                atl08.df['DW'] = da.sel(band=1).interp(
                    y=("points", atl08.df.latitude.values),
                    x=("points", atl08.df.longitude.values),
                    method="nearest"
                ).values
                atl08.df = atl08.df[~atl08.df['DW'].isin([0])]

            if landcover != None:
                atl08.df = atl08.df[atl08.df['segment_landcover'].isin([111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126])]

            atl08.df = atl08.df[abs(atl08.df['h_te_interp'] - altitude) <= 80]
            atl08.df = atl08.df[(atl08.df['layer_flag'] < 1)|(atl08.df['msw_flag']<1)]

            atl08.df = atl08.df.rename(columns={'photon_rate_te': 'Eg', 'photon_rate_can_nr': 'Ev'})

            if sat_flag != 0:
                atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

            # Dataframe of the latitudes, longitudes, and Ev/Eg depending on parameter
            df = atl08.df.loc[:,['latitude','longitude', c]]
            if big_df.shape[0] == 0:
                big_df = df
            else:
                big_df = pd.concat([big_df, df], ignore_index = True)

            

    
    return big_df
    
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