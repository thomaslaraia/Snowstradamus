import os
import geopandas as gpd
from shapely.geometry import Point, box as shapely_box
import simplekml
import numpy as np
import zipfile
import shutil

def make_shapefile(string, coords, width_km=5, height_km=5, output_format='GeoJSON'):

    # Approximate conversions
    km_per_degree_lat = 111  # Kilometers per degree of latitude
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))  # Kilometers per degree of longitude at given latitude

    # Convert width and height from kilometers to degrees
    width = width_km / km_per_degree_lon
    height = height_km / km_per_degree_lat
    
    # Create folders for output if they don't exist
    output_folders = ['../data_store/data/shapefiles', './shapefiles']
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)

    # Create a GeoDataFrame with a point and set the CRS explicitly
    point = gpd.GeoDataFrame(geometry=[Point(coords)], crs="EPSG:4326")

    # Create a square (box) with a given width and height
    bounding_box = shapely_box(
        point.geometry.x[0] - width, 
        point.geometry.y[0] - height,
        point.geometry.x[0] + width, 
        point.geometry.y[0] + height
    )

    # Convert the box to a GeoDataFrame
    box_gdf = gpd.GeoDataFrame(geometry=[bounding_box], crs="EPSG:4326")

    # Save the file in the specified format
    for output_folder in output_folders:
        if output_format.lower() == 'geojson':
            filename = os.path.join(output_folder, f"{string}.geojson")
            box_gdf.to_file(filename, driver='GeoJSON')
            print(f"GeoJSON file '{filename}' created.")
        
        elif output_format.lower() == 'shapefile':
            # Create a temporary folder for shapefile components
            shapefile_folder = os.path.join(output_folder, string)
            os.makedirs(shapefile_folder, exist_ok=True)
            
            # Define the path for the shapefile components
            filename = os.path.join(shapefile_folder, f"{string}.shp")
            box_gdf.to_file(filename)
            
            # Zip the shapefile components
            zip_filename = os.path.join(output_folder, f"{string}.zip")
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in os.listdir(shapefile_folder):
                    zipf.write(os.path.join(shapefile_folder, file), arcname=file)
            
            # Remove the temporary shapefile folder
            shutil.rmtree(shapefile_folder)
            print(f"Shapefile '{zip_filename}' created and zipped.")
        
        elif output_format.lower() == 'kml':
            # Create a KML object
            kml = simplekml.Kml()
            
            # Create a polygon from the box geometry
            polygon_coords = [(x, y) for x, y in box_gdf.geometry.apply(lambda geom: list(geom.exterior.coords))[0]]
            kml.newpolygon(name=f"Box around {string}", outerboundaryis=polygon_coords)
            
            filename = os.path.join(output_folder, f"{string}.kml")
            kml.save(filename)
            print(f"KML file '{filename}' created.")
        
        else:
            print(f"Unsupported format: {output_format}")
    
    return

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, help="name of shapefile you want to create")
    parser.add_argument("--lon", type=float, help="longitude of location of interest")
    parser.add_argument("--lat", type=float, help="latitude of location of interest")
    parser.add_argument("--w", type=float, default=4, help="distance from center point to box side")
    parser.add_argument("--h", type=float, default=4, help="distance from center point to top/bottom")
    parser.add_argument("--format", type=str, default="geojson", help="output format of shapefile, can do kml, geojson, shapefile. That's it.")

    args = parser.parse_args()

    make_shapefile(string = args.name,
                   coords = (args.lon, args.lat),
                   width_km = args.w,
                   height_km = args.h,
                   output_format = args.format)
