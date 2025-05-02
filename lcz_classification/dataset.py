
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
        


def ee_download(asset_id, bands, date_range, bbox, output_dir, tile_dims=None, scale=30):
    """Download ALOS DSM data for a specified bounding box using Google Earth Engine.

    Args: 
        asset_id (str): Asset ID of dataset on Google Earth Engine Catalog
        bands (list): Desired bands to select and download from the requested dataset
        date_range (list): start and end date to filter the ee.ImageCollection
        bbox (list): Bounding box coordinates in format [west, south, east, north]
        output_dir (str): Output directory to save downloaded GeoTIFFs
        tile_dims (tuple): Dimensions of tiling for subdividing the bbox into smaller regions, useful for downloading from large areas
        scale (int): Desired spatial resolution in meters
        
    Returns:
        str: Path to the downloaded GeoTIFF file
    """
    # Initialize Earth Engine
    try:
   
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
        print("Error initializing Earth Engine. Make sure you're authenticated.")
        print(f"Error details: {e}")
        print("Run 'earthengine authenticate' in terminal if you haven't authenticated.")
        return None
    

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    images= list()
    
    start_date, end_date=date_range
    band_list =list()
    # try:
    # Create a bounding box geometry from coordinates
    geometry = ee.Geometry.Rectangle(bbox)
    
    data_type=ee.data.getInfo(asset_id)['type'] # Retrieve data type based on asset_id
    print(data_type)

    if data_type == "IMAGE_COLLECTION":

        # # Load the ImageCollection
        # image_collection = ee.ImageCollection(asset_id).filterBounds(geometry).filterDate(start_date,end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))  # Filter for less than 20% cloud cover
        
        date=ee.Date(start_date)

        search=True
        while search:
            m=ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(geometry).filterDate(date, date.advance(1,unit='day')).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

            features=m.getInfo()['features']
            feature_count=len(features)
            print(f'{feature_count} Features found')
            if feature_count > 0:
                search=False
                print(f'Found Collection on Date {date.getInfo()}')
                        # Iterate over features within the image collection
                for feature in features:
                    
                    image_id=feature['id']
                    image=ee.Image(image_id)
                    # Iterate over bands
                    for band in bands:
                        
                        band_image=image.select(band)
                        images.append(band_image)
                
                        band_list.append(band)
            else:
                print('Advancing date 1 day')
                date=date.advance(1,unit='day')

    

    elif data_type == "IMAGE":
            image=ee.Image(asset_id).select(band)
            # Iterate over bands
            for band in bands:
                 # Iterate over bands
                band_image=image.select(band)
                image_id = image.getInfo()['id']
                images.append(band_image)
                band_list.append(band)
            
    for im,b in zip(images,band_list):
    
        im_id=im.getInfo()['id'].split("/")[-1].split("_")[-1]
        im_dir=f"{output_dir}/{im_id}"
        if os.path.exists(im_dir) == False:
            os.mkdir(im_dir)
        
        if tile_dims is not None:
            

            tile_bbox=im.geometry().bounds().intersection(ee.Geometry.Rectangle(bbox) ,maxError=1)
            tile_bbox_coords=tile_bbox.bounds().getInfo()['coordinates']
            xx, yy  = np.array(tile_bbox_coords).T
            in_bbox=[min(xx), min(yy), max(xx), max(yy)]
            # return tile_bbox_coords
            tiles=tiles_from_bbox(bbox=in_bbox,tile_dims=tile_dims)
            for idx, tile in tiles.iterrows():
                tile_path=f"{im_dir}/{im_id}_{b}_{tile.tile_id}.tif"
                if os.path.exists(tile_path) == False:
                  
                    print(f'Downloading {im_id} tile {tile.tile_id}')
                    clipped=im.clip(ee.Geometry.Rectangle(list(tile.geometry.bounds)))
                    geemap.ee_export_image(
                        clipped,
                        filename=tile_path,
                        scale=scale,
                        file_per_band=False,
                        crs='EPSG:4326', 
                        region = list(tile.geometry.bounds)
                )
        else:

            print(f'Downloading {im_id}')
            clipped=im.clip(ee.Geometry.Rectangle(bbox))
            im_dir=f"{output_dir}/{im_id}"
            os.mkdir(im_dir)
            geemap.ee_export_image(
                clipped,
                filename=f"{im_dir}/{im_id}_{b}.tif",
                scale=scale,
                file_per_band=False,
                crs='EPSG:4326', 
                region = bbox
            )

    

    print("EE DOWNLOAD COMPLETE")



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
     
        