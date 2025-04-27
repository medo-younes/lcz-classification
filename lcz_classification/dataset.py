import osmnx as ox
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import geopandas as gpd
import arrow
import json
import os
from lcz_classification.util import tiles_from_bbox
import ee
import geemap
import osmnx as ox
import pandas as pd
from lcz_classification.config import STUDY_AREA_FP, CRS, S2_METADATA_FP, S2_TILES_FP, LCZ_LEGEND_FP
import numpy as np
from shapely.geometry import box
# AWS S3 DATA DOWNLOAD
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))



def fetch_metadata(name):

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
        



def get_matching_s3_keys(bucket, prefix='', suffix=''):
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.endswith(suffix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break



def get_matching_scenes(bucket, tiles, date_range, prefix):
    matching_scenes = []

    for tile in tiles:
        keys = list(get_matching_s3_keys(bucket, prefix=f'{prefix}/{tile[0:2]}/{tile[2]}/{tile[3:5]}/'))
        for key in keys:
            scene_id = key.split('/')[-2]
            scene_date = arrow.get(scene_id.split('_')[2], 'YYYYMMDD')
            if scene_id not in matching_scenes and scene_date >= date_range[0] and scene_date <= date_range[1]:
                matching_scenes.append(scene_id)

    
    print(f'Found {len(matching_scenes)} matching scenes')
    print(matching_scenes)
    return matching_scenes


def download_tiles(scenes, bands, prefix, out_dir):
    
    for scene in scenes:
        scene_dir=f'{out_dir}/{scene}'
        print(f"- Sentinel-2 -- Downloading Bands {', '.join(bands)} from Scene {scene}")
        tile = scene.split('_')[1]
        date = scene.split('_')[2]
        year = date[0:4]
        month = date[4:6]
        
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        
            files=[f"{b}.tif" for b in bands]
      
            for f in files:
                key = f'{prefix}/{tile[0:2]}/{tile[2]}/{tile[3:5]}/{year}/{int(month)}/{scene}/{f}'
            
                fname = key.split('/')[-1]
                print(f'{scene_dir}/{fname}')
                s3.download_file('sentinel-cogs', key, f'{scene_dir}/{fname}')

        else:
            print(f"{scene_dir} already exists")






def ee_download(asset_id, bands, date_range, bbox, output_dir, tile_dims=None, scale=30):
    """
    Download ALOS DSM data for a specified bounding box using Google Earth Engine.
    
    Parameters:
    -----------
    bbox : list
        Bounding box coordinates in format [west, south, east, north]
    output_dir : str
        Directory to save the downloaded data
    filename : str
        Output filename for the GeoTIFF
    scale : int
        Resolution in meters for the exported image (default is 30m to match ALOS)
    
    Returns:
    --------
    str
        Path to the downloaded GeoTIFF file
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



def get_city_polygon(city,country):
    # Get city boundary polygon
    gdf = ox.geocode_to_gdf(f"{city}, {country}")
    gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)


    # Plot the boundary
    gdf.explore(style={
                    "fill": False,
                    "color": "red"
    })

    return gdf
     
        