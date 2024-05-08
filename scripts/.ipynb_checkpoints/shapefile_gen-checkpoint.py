import geopandas as gpd
from shapely.geometry import Point, box as shapely_box
import simplekml
import os


def make_shapefile(string, coords, width=4000, height=4000):

    # Create a folder named 'shapefiles'
    output_folder = 'shapefiles'
    os.makedirs(output_folder, exist_ok=True)

    # Create a GeoDataFrame with a coordinate point and set the CRS explicitly
    point = gpd.GeoDataFrame(geometry=[Point(coords)], crs="EPSG:4326")

    # Reproject to a projected CRS (e.g., UTM)
    point_utm = point.to_crs(epsg=32630)  # You may need to choose an appropriate EPSG code

    # Create a square (box) with a given width and height
    bounding_box = shapely_box(point_utm.geometry.x[0] - width / 2, point_utm.geometry.y[0] - height / 2,
                               point_utm.geometry.x[0] + width / 2, point_utm.geometry.y[0] + height / 2)

    # Convert the box to a GeoDataFrame
    box_gdf = gpd.GeoDataFrame(geometry=[bounding_box], crs=point_utm.crs)

    # Reproject back to the original CRS
    box = box_gdf.to_crs(epsg=4326)

    # Create a subfolder for each string
    string_folder = os.path.join(output_folder, f"{string}")
    os.makedirs(string_folder, exist_ok=True)

    # Create a KML object
    kml = simplekml.Kml()

    # Create a polygon from the box geometry
    polygon_coords = [(x, y) for x, y in box.geometry.apply(lambda geom: list(geom.exterior.coords))[0]]
    polygon = kml.newpolygon(name=f"Box around {string}", outerboundaryis=polygon_coords)

    # Save the KML file in the subfolder
    kml_filename = os.path.join(string_folder, f"{string}_box.kml")
    kml.save(kml_filename)

    print(f"KML file '{kml_filename}' created.")
    return
