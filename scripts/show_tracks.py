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
import matplotlib.patches as mpatches

def plot_static_map_with_box(df, coords, c='Eg', cmap='viridis', vmin=0, vmax=6,
                             save='no', w=4, h=4, cam = 'Unknown', date_str = 'unknown_date',
                             km_boxes=True, km_box_beams=(0, 2, 4)):
    lon_center, lat_center = coords
    km_to_deg_lat = 1 / 111
    km_to_deg_lon = 1 / (111 * np.cos(np.radians(lat_center)))

    half_box_deg_lat = h * km_to_deg_lat
    half_box_deg_lon = w * km_to_deg_lon
    pad_deg_lat = h * 0.2 * km_to_deg_lat
    pad_deg_lon = w * 0.2 * km_to_deg_lon

    box_geom = box(lon_center - half_box_deg_lon,
                   lat_center - half_box_deg_lat,
                   lon_center + half_box_deg_lon,
                   lat_center + half_box_deg_lat)

    extent = [lon_center - half_box_deg_lon - pad_deg_lon,
              lon_center + half_box_deg_lon + pad_deg_lon,
              lat_center - half_box_deg_lat - pad_deg_lat,
              lat_center + half_box_deg_lat + pad_deg_lat]

    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                           crs="EPSG:4326")
    gdf_web = gdf.to_crs(epsg=3857)
    box_web = gpd.GeoSeries([box_geom], crs="EPSG:4326").to_crs(epsg=3857)

    extent_web = gpd.GeoSeries([Point(x, y) for x, y in [
        (extent[0], extent[2]), (extent[1], extent[3])
    ]], crs="EPSG:4326").to_crs(epsg=3857).total_bounds

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_web.plot(ax=ax, column=c, cmap=cmap, markersize=3,
                 legend=True, vmin=vmin, vmax=vmax)

    cbar = ax.get_figure().get_axes()[-1]
    cbar.set_ylabel("Ground Radiometry", rotation=270, labelpad=15, fontsize=13)
    cbar.tick_params(labelsize=13)

    # Red 8x8 km box (your existing region)
    box_web.boundary.plot(ax=ax, color='red', linewidth=1.5, linestyle=':')

    # -----------------------------
    # ADD 1x1 km YELLOW BOXES (8 latitude bands, correct km sizing)
    # -----------------------------
    if km_boxes and ('beam' in gdf_web.columns):
    
        # Red box bounds in *degrees* (EPSG:4326)
        min_lon, min_lat, max_lon, max_lat = box_geom.bounds  # NOTE: box_geom is in EPSG:4326
    
        # 8 equal latitude bands spanning the whole red box
        lat_edges = np.linspace(max_lat, min_lat, 9)  # 9 edges -> 8 bands
    
        for beam_id in km_box_beams:
            # Use EPSG:4326 points for "is there data in this latitude band?"
            pts_ll = gdf[gdf['beam'] == beam_id].copy()
            if pts_ll.shape[0] < 5:
                continue
    
            # Restrict to the red box (in degrees)
            in_red = (
                (pts_ll['longitude'].values >= min_lon) & (pts_ll['longitude'].values <= max_lon) &
                (pts_ll['latitude'].values  >= min_lat) & (pts_ll['latitude'].values  <= max_lat)
            )
            pts_ll = pts_ll.loc[in_red]
            if pts_ll.shape[0] < 5:
                continue
    
            lons = pts_ll['longitude'].to_numpy()
            lats = pts_ll['latitude'].to_numpy()
    
            # Fit groundtrack in degrees: lon = m*lat + b
            m, b = np.polyfit(lats, lons, 1)
    
            for lat_top, lat_bot in zip(lat_edges[:-1], lat_edges[1:]):
                lat_c = 0.5 * (lat_top + lat_bot)
    
                # Does this beam have any data in this latitude slice?
                in_band = (lats <= lat_top) & (lats > lat_bot)  # top->bottom band
                if not np.any(in_band):
                    continue
    
                # Centre longitude on the fitted track
                lon_c = m * lat_c + b
    
                # 1 km in degrees at this latitude
                dlat = 1.0 / 111.0
                dlon = 1.0 / (111.0 * np.cos(np.radians(lat_c)))
    
                # Build a 1x1 km box in EPSG:4326 (degrees)
                box_ll = box(
                    lon_c - 0.5 * dlon, lat_c - 0.5 * dlat,
                    lon_c + 0.5 * dlon, lat_c + 0.5 * dlat
                )
    
                # Clip the small box to the big 8x8 km box (both EPSG:4326)
                clipped_ll = box_ll.intersection(box_geom)
                
                # If nothing overlaps, skip
                if clipped_ll.is_empty:
                    continue
                
                # Reproject the clipped geometry to EPSG:3857 for plotting
                clipped_web = gpd.GeoSeries([clipped_ll], crs="EPSG:4326").to_crs(epsg=3857)
                
                # Plot (handles partial boxes at edges)
                clipped_web.boundary.plot(ax=ax, color='yellow', linewidth=2)
    
                # Reproject the box to EPSG:3857 for plotting on your current axes
                # box_web = gpd.GeoSeries([box_ll], crs="EPSG:4326").to_crs(epsg=3857)
    
                # box_web.boundary.plot(ax=ax, color='yellow', linewidth=2)
    # -----------------------------

    ax.set_xlim(extent_web[0], extent_web[2])
    ax.set_ylim(extent_web[1], extent_web[3])
    ax.set_aspect('equal', adjustable='box')
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def format_lon(x, _):
        lon, _ = transformer.transform(x, 0)
        return f"{lon:.2f}°"

    def format_lat(y, _):
        _, lat = transformer.transform(0, y)
        return f"{lat:.2f}°"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)

    plt.title(f"Ground radiometry within 8×8 km box around {cam} on {date_str}",
             fontsize=15)
    plt.tight_layout()
    if save != 'no':
        plt.savefig(save, dpi=300)
    plt.show()

def plot_tracks_on_dynamicworld(df, coords, dw_name=None, dw_dir='../scratch/data/DW',
                                c='Eg', cmap='viridis', vmin=None, vmax=None,
                                w=4, h=4, save='no'):
    """
    Plots ATL08 points over DynamicWorld landcover in EPSG:4326 (lat/lon).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import rioxarray
    import geopandas as gpd
    from shapely.geometry import box
    import numpy as np

    # Center and bounding box in degrees
    lon_center, lat_center = coords
    km_to_deg_lat = 1 / 111
    km_to_deg_lon = 1 / (111 * np.cos(np.radians(lat_center)))

    half_box_deg_lat = h * km_to_deg_lat
    half_box_deg_lon = w * km_to_deg_lon
    pad_deg_lat = h * 0.2 * km_to_deg_lat
    pad_deg_lon = w * 0.2 * km_to_deg_lon

    core_box = box(lon_center - half_box_deg_lon,
                   lat_center - half_box_deg_lat,
                   lon_center + half_box_deg_lon,
                   lat_center + half_box_deg_lat)

    extent = [lon_center - half_box_deg_lon - pad_deg_lon,
              lon_center + half_box_deg_lon + pad_deg_lon,
              lat_center - half_box_deg_lat - pad_deg_lat,
              lat_center + half_box_deg_lat + pad_deg_lat]

    padded_box = box(*extent)

    # Load DW raster (no reprojection)
    if dw_name is None and 'camera' in df.columns:
        dw_name = df['camera'].iloc[0]
    filepath = find_dynamicworld_file(dw_name, dw_dir)
    da = rioxarray.open_rasterio(filepath, masked=True)

    # Clip DW to extent in EPSG:4326
    dw_clipped = da.rio.clip_box(minx=extent[0], maxx=extent[1],
                                  miny=extent[2], maxy=extent[3])

    # Filter and convert ICESat-2 points
    df = df[(df['longitude'] >= extent[0]) & (df['longitude'] <= extent[1]) &
            (df['latitude'] >= extent[2]) & (df['latitude'] <= extent[3])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    dw_clipped.plot(ax=ax, cmap='tab20', add_colorbar=False, alpha=0.7)

    if not gdf.empty:
        if vmin is None:
            vmin = gdf[c].min()
        if vmax is None:
            vmax = gdf[c].max()
        gdf.plot(ax=ax, column=c, cmap=cmap, markersize=5, legend=True, vmin=vmin, vmax=vmax)

    gpd.GeoSeries(core_box, crs="EPSG:4326").boundary.plot(ax=ax, edgecolor='red', linewidth=1.5, linestyle='--')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))
    
    ax.set_title(f"{c} over DynamicWorld around ({lat_center:.4f}, {lon_center:.4f})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    from matplotlib.patches import Patch
    import matplotlib.cm as cm

    cmap_tab20 = cm.get_cmap('tab20', 10)
    present_classes = np.unique(dw_clipped.values[~np.isnan(dw_clipped.values)]).astype(int)

    legend_elements = [
        Patch(facecolor=cmap_tab20(i), edgecolor='k', label=f"Class {i}")
        for i in present_classes if 0 <= i < 10
    ]

    ax.legend(handles=legend_elements, title="DW Class", loc='lower right', fontsize=8, title_fontsize=9)
    
    plt.tight_layout()

    if save != 'no':
        plt.savefig(save, dpi=300)
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

def show_tracks(dirpath, atl03paths, atl08paths, coords, altitude, c = 'Eg', gtx = None, CBAR = None, w=4, h=4, landcover=None,
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

    foldername = dirpath.split('/')[-2]
    
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

        
        for i, gt in enumerate(tracks):
            
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
                filepath = find_dynamicworld_file(foldername)
                da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")
    
                if atl08.df.shape[0] == 0:
                    # Ensure the DW column exists even if there are no rows,
                    # and *skip* the expensive interpolation that would fail on empty coords.
                    atl08.df['DW'] = np.array([], dtype='float32')
                else:
                    atl08.df['DW'] = da.sel(band=1).interp(
                        y=("points", atl08.df.latitude.values),
                        x=("points", atl08.df.longitude.values),
                        method="nearest"
                    ).values
                    atl08.df = atl08.df[atl08.df['DW'] == 1]

            if landcover != None:
                atl08.df = atl08.df[atl08.df['segment_landcover'].isin([111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126])]

            atl08.df = atl08.df[abs(atl08.df['h_te_interp'] - altitude) <= 80]
            atl08.df = atl08.df[(atl08.df['layer_flag'] < 1)|(atl08.df['msw_flag']<1)]

            atl08.df = atl08.df.rename(columns={'photon_rate_te': 'Eg', 'photon_rate_can_nr': 'Ev'})
            if i + 1 == 3:
                atl08.df.Eg /= 0.85
                atl08.df.Ev /= 0.85

            if sat_flag != 0:
                atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

            # Dataframe of the latitudes, longitudes, and Ev/Eg depending on parameter
            df = atl08.df.loc[:,['latitude','longitude', c]]
            df['beam'] = i
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