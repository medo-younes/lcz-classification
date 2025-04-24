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


# AWS S3 DATA DOWNLOAD
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


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






def ee_download(asset_id, bands, bbox, output_dir, tile_dims=None, scale=30):
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
    
    
    # try:
    # Create a bounding box geometry from coordinates
    geometry = ee.Geometry.Rectangle(bbox)
    
    data_type=ee.data.getInfo(asset_id)['type'] # Retrieve data type based on asset_id
    print(data_type)

    if data_type == "IMAGE_COLLECTION":

        # Load the ImageCollection
        image_collection = ee.ImageCollection(asset_id).filterBounds(geometry)

        # Iterate over features within the image collection
        for feature in image_collection.getInfo()["features"]:
            # Iterate over bands
            for band in bands:
                image=ee.Image(feature["id"]).select(band).clip(geometry)
                output_path=f"{output_dir}/{feature['id'].split('/')[-1]}_{band}.tif"
                print(f"Downloading Band {band} from Image {feature['id']} in ImageCollection {asset_id}")
                # Use geemap for downloading (handles GEE export process)
                geemap.ee_export_image(
                    image,
                    filename=output_path,
                    scale=scale,
                    region=geometry,
                    file_per_band=False,
                    crs='EPSG:4326'
                )
    elif data_type == "IMAGE":
        
        # Downloading Data wihout Tiling
        if tile_dims is None:   

            # Iterate over bands
            for band in bands:
                image=ee.Image(asset_id).select(band).clip(geometry)
                image=image.reproject(crs=image.projection(), scale=scale)
                output_path=f"{output_dir}/{asset_id.split('/')[-1]}_{band}.tif"
                print(f"Downloading {band} from {asset_id}")
                # Use geemap for downloading (handles GEE export process)
                geemap.ee_export_image(
                    image,
                    filename=output_path,
                    scale=scale,
                    region=geometry,
                    file_per_band=False,
                    crs='EPSG:4326'
                )

        #Downloading data with Tiling
        else:
            tiles=tiles_from_bbox(bbox=bbox,tile_dims=tile_dims)
            for idx,tile in tiles.iterrows():
                geometry=ee.Geometry.Rectangle(list(tile.geometry.bounds))
                
                # Iterate over bands
                for band in bands:
                    image=ee.Image(asset_id).select(band).clip(geometry)
                    image=image.reproject(crs=image.projection(), scale=scale)
                    output_path=f"{output_dir}/{asset_id.split('/')[-1]}_{band}_{tile.tile_id}.tif"
                    print(f"Downloading {band} from {asset_id}")
                    # Use geemap for downloading (handles GEE export process)
                    geemap.ee_export_image(
                        image,
                        filename=output_path,
                        scale=scale,
                        region=geometry,
                        file_per_band=False,
                        crs='EPSG:4326'
                    )

            tiles.to_file(f"{output_dir}/tiles.geojson")

        
        