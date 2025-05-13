
from lcz_classification.config import STUDY_AREA_FP, CRS, S2_METADATA_FP, S2_TILES_FP, LCZ_LEGEND_FP


import geopandas as gpd
import os
from lcz_classification.util import tiles_from_bbox
import ee
import geemap
import osmnx as ox
import pandas as pd
import numpy as np



def fetch_metadata(name):
    """Retrieve metadata dataset using filepaths configured in config.py
    
    Args:
        name (str): Name of metadata to fetch 1 of 4 options: 
                    (1) STUDY_AREA
                    (2) S2_TILES
                    (3) S2_METADATA
                    (4) LCZ_LEGEND
    
    Returns:
        DataFrame: Desired metadata in a DataFrame or GeoDataFrame object

    """

    d=dict(
        STUDY_AREA = dict(fp=STUDY_AREA_FP, tp = 'gdf'),
        S2_TILES = dict(fp=S2_TILES_FP, tp = 'gdf'),
        S2_METADATA = dict(fp=S2_METADATA_FP, tp = 'csv'),
        LCZ_LEGEND = dict(fp=LCZ_LEGEND_FP, tp = 'csv'),
    )
    
    # Get file path and data type
    fp = d[name]['fp']
    tp = d[name]['tp']


    if tp == 'gdf':
        # STUDY AREA
        return gpd.read_file(fp).to_crs(CRS) # Read study area bounds as GeoDataFrame

    elif tp == 'csv':
        return pd.read_csv(fp)
        


import geemap

def ee_get_image(col_id, band,bbox):
    
    geom=ee.Geometry.Rectangle(bbox)

    asset_type=ee.data.getInfo(col_id)['type']

    if asset_type == 'IMAGE_COLLECTION':
        image=ee.ImageCollection(col_id).filterBounds(geom).select(band).mean().clip(geom)
    else:
        image=ee.Image(col_id).select(band).clip(geom)

    image_id = col_id.replace('/','_') + f'_{band}'
   
    return image, image_id


def ee_download_tiled_image(image, image_id,tiles_gdf, scale, output_dir,):

    for idx, tile in tiles_gdf.iterrows():
        tile_id=tile.tile_id
        bbox=tile.geometry.bounds
        bbox_geom=ee.Geometry.Rectangle(bbox)
        filename=f"{output_dir}/{image_id}_{tile_id}_{scale}m.tif"
        print(filename)
        if os.path.exists(filename) == False:

            geemap.ee_export_image(
                image,
                filename=filename,
                scale=scale,
                file_per_band=False,
                crs='EPSG:4326', 
                region = bbox_geom
            )
    
            print(f"Downloaded Image: {image_id}_{tile_id}_{scale}m.tif")
        else:
            print(f"{filename} Already Exists")

def get_city_polygon(city:str,country:str) -> gpd.GeoDataFrame:
    """
    Using osmnx OpenStreetMap library, get polygon of city boundaries as a GeoDataFrame.
    
    Args:
        city (str): The city you want a polygon of
        country (str): The country of your city
    
    Returns:
        GeoDataFrame: Polygon geometry of city boundary.

    """
    # Get city boundary polygon
    gdf = ox.geocode_to_gdf(f"{city}, {country}")
    gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)


    # Plot the boundary
    gdf.explore(style={
                    "fill": False,
                    "color": "red"
    })

    return gdf
     
        