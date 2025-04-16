import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import shapely
import math


def lon_lat_to_utm_epsg_code(lon, lat):
    """
    Function to retrieve local UTM EPSG code from WGS84 geographic coordinates.
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code

def EE_image_corner_coords_to_polygon_gdf(df,
                                          epsg_code='4326'):
    
    df_tmp = df.copy() # avoid modifying original
    
    lat_keys = ['NWlat', 'NElat', 'SElat', 'SWlat']
    lon_keys = ['NWlon', 'NElon', 'SElon', 'SWlon']
    
    polygons = []

    for index, row in df_tmp.iterrows():
        lons = list(row[lon_keys].values)
        lats = list(row[lat_keys].values)
        lons.append(row[lon_keys].values[0])
        lats.append(row[lat_keys].values[0])
        coords = list(zip(lons,lats))
        polygon = Polygon(coords)
        
        if lons[0] == lons[1]: #handles EarthExplorer instances where corner coordinates are identical
            polygon= gpd.GeoDataFrame(geometry=[polygon,],
                                      crs='epsg:'+'4326').to_crs('epsg:'+ epsg_code)['geometry'].buffer(0.018).to_crs('epsg:'+'4326').iloc[0]

        polygons.append(polygon)
        
    gdf = gpd.GeoDataFrame(df_tmp,
                           geometry=polygons,
                           crs='epsg:'+'4326').to_crs('epsg:'+ epsg_code)
    return gdf


# From https://gis.stackexchange.com/questions/387773/count-overlapping-features-using-geopandas
def count_overlapping_features(in_gdf):
    # Get the name of the column containing the geometries
    geom_col = in_gdf.geometry.name
    
    # Setting up a single piece that will be split later
    input_parts = [in_gdf.unary_union.buffer(0)]
    
    # Finding all the "cutting" boundaries. Note: if the input GDF has 
    # MultiPolygons, it will treat each of the geometry's parts as individual
    # pieces.
    cutting_boundaries = []
    for i, row in in_gdf.iterrows():
        this_row_geom = row[geom_col]
        this_row_boundary = this_row_geom.boundary
        if this_row_boundary.type[:len('multi')].lower() == 'multi':
            cutting_boundaries = cutting_boundaries + list(this_row_boundary.geoms)
        else:
            cutting_boundaries.append(this_row_boundary)
    
    
    # Split the big input geometry using each and every cutting boundary
    for boundary in cutting_boundaries:
        splitting_results = []
        for j,part in enumerate(input_parts):
            new_parts = list(shapely.ops.split(part, boundary).geoms)
            splitting_results = splitting_results + new_parts
        input_parts = splitting_results
    
    # After generating all of the split pieces, create a new GeoDataFrame
    new_gdf = gpd.GeoDataFrame({'id':range(len(splitting_results)),
                                geom_col:splitting_results,
                                },
                               crs=in_gdf.crs,
                               geometry=geom_col)
    
    # Find the new centroids.
    new_gdf['geom_centroid'] = new_gdf.centroid
    
    # Starting the count at zero
    new_gdf['count_intersections'] = 0
    
    # For each of the `new_gdf`'s rows, find how many overlapping features 
    # there are from the input GDF.
    for i,row in new_gdf.iterrows():
        new_gdf.loc[i,'count_intersections'] = in_gdf.intersects(row['geom_centroid']).astype(int).sum()
        pass
    
    # Dropping the column containing the centroids
    new_gdf = new_gdf.drop(columns=['geom_centroid'])[['id','count_intersections',geom_col]]
    
    return new_gdf