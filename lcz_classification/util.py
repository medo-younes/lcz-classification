

import numpy as np
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
import string
import rasterio as r
from rasterio.merge import merge
from rasterio.enums import Resampling
import pandas as pd
import xarray as xr
from rasterstats import zonal_stats
import math
from prettytable import PrettyTable
from pyproj.crs import CRS as pycrs
import fiona 

import math


fiona.supported_drivers['KML'] = 'r'  # Explicitly enable KML read support

def kml_to_gdf(kml_path):
    """Read KML file a GeoPandas GeoDataFrame
    
    Args:
        kml_path (str): Path to KML file


    Returns:
        GeoDataFrame: Geometry features of KML file read into GeoPandas GeoDataFrame
    """
    layers = fiona.listlayers(kml_path)
    gdf_list = []

    for layer in layers:
        try:
            with fiona.open(kml_path, driver='KML', layer=layer) as src:
                gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
                gdf_list.append(gdf)
        except:
            print(f'Non-valid Layer: {layer}')

    # Combine all into one GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    return combined_gdf


def normalize_arr(arr, scaler):
    """Rescale values of an array using a given scaler such as MinMaxScaler
    
    Args:
        arr (np.array): Numpy array
        scaler (sklearn.preprocessing.MinMaxScaler): scaler object from sklearn.preprocessing

    Returns:
        np.array: Rescaled numpy array
    """
    return scaler.fit_transform(arr.reshape(-1,1)).reshape(arr.shape)


def read_lcz_legend(file_path):
    lcz_legend=pd.read_csv(file_path)
    lcz_legend["hex"]=lcz_legend.apply(lambda row: '#%02x%02x%02x' % (row.r,row.g,row.b) , axis=1)
    lcz_legend['rgb'] = lcz_legend.apply(lambda x: (x["r"],  x["g"], x["b"]), axis = 1)
    color_dict=lcz_legend[["name", "hex"]].set_index("name").hex.to_dict()

    return lcz_legend, color_dict


def tiles_from_bbox(bbox, crs=4236, tile_dims=(4,4)):
    """Generates polygon tiles subdividing a given bounding box based on the given dimensions
    
    Args:
        bbox (list): Bounding box coordinates in format [x1,y1,x2,y2]
        tile_dims (tuple): How many tiles to sub-section the bounding box with the dimensions (width, length). With tile_dims == (4,4) the function will return 16 (4 x 4) tile polygons

    Returns:
        GeoDataFrame: GeoPandas GeoDataFrame of generated tiles as Shapely Polygons
    """

    x1,y1,x2,y2=bbox # Get left, top, right, bottom from bounding box

    x_rg = x2 - x1
    y_rg = y2 - y1

    x_inc=x_rg / (tile_dims[0])
    y_inc=y_rg / (tile_dims[1])


    xx=np.arange(x1, x2, x_inc)
    xx=np.append(xx,x2)
    yy=np.arange(y1, y2, y_inc)
    yy=np.append(yy,y2)

    xx,yy=np.meshgrid(xx, yy, indexing='ij')
    yy=yy.transpose()
 
    xx1=xx[:-1]
    xx2=xx[1:]
    yy1=yy[:-1]
    yy2=yy[1:]

    bounds=list()
    

    for lon1, lon2 in zip(xx1, xx2):    
        for lat1,lat2 in zip(yy1,yy2):
            bounds.extend([box(*(x1,y1,x2,y2)) for x1,y1,x2,y2 in zip(lon1,lat1,lon2,lat2)])
       

    tiles=gpd.GeoDataFrame( geometry=bounds, crs=crs).drop_duplicates("geometry").reset_index(drop=True)

    letters = list(string.ascii_uppercase)[0:tile_dims[0]]
    numbers = list(range(1, tile_dims[1] + 1))
    numbers.reverse()

    tiles["tile_id"]=[f"{l}{n}"  for l in letters for n in numbers]
    return tiles.sort_values('tile_id')


def merge_rasters(raster_paths, out_path, transform):
    """Merge multiple rasters into one
    
    Args:
        raster_paths (list): Paths of rasters to merge
        out_path (list): File path of merged raster
        transform (rasterio.Affine): Affine transform of output raster, retrieved from a reference raster


    Returns:
        None
    """
    rasters= [r.open(path) for path in raster_paths]

    
    array, merge_transform = merge(rasters)

    if transform is not None:
        out_transform = transform
    else: 
        out_transform = merge_transform
    # Use metadata from one of the source files and update it
    
    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "height": array.shape[1],
        "width": array.shape[2],
        "transform": out_transform
    })

    # Write the merged output
    with r.open(out_path, "w", **out_meta) as dest:
        dest.write(array)
    


def get_target_shape(raster,target_res):
    '''Resample an xarray DataArray to a targett resolution
    
    Args
        raster (xr.DataArray): Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
        target_res (int): Cell resolution in METERS that you want to resample the raster to

    Returns
        tuple: Desired shape of raster when resampling to target resolution
    
    '''
    if target_res > 0.0099 and target_res <= 0.99:
        print(f"Resampling input raster to {target_res * 100} cm resolution")
    elif  target_res <= 0.0099:
        print(f"Resampling input raster to {target_res * 1000} mm resolution")
    else:
        print(f"Resampling input raster to {target_res} m resolution")

    orig_res=raster.rio.resolution() # Get original resolution of raster
    orig_res_x = orig_res[0] # resolution in x / lon direction
    orig_res_y = abs(orig_res[1]) # resolution in y / lat direction

    orig_width = raster.rio.width # original width
    orig_height = raster.rio.height # original height


    rescale_x=target_res / orig_res_x # X rescale factor
    rescale_y=target_res / orig_res_y # Y rescale factor

    target_width= round(orig_width / rescale_x) # Calculate Target Width
    target_height=round(orig_height / rescale_y) # Calculate Target Height


    return (target_height, target_width)

def resample_da(raster,target_res, resampling=Resampling.bilinear):
    '''Resample an xarray DataArray to a targett resolution
    
    Args
        raster (xr.DataArray): Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
        target_res (int): Cell resolution in METERS that you want to resample the raster to

    Returns
        rescaled (xr.DataArray): Raster rescaled to the target resolution, keeping the original projection of the input raster
    
    '''

    if target_res > 0.0099 and target_res <= 0.99:
        print(f"Resampling input raster to {target_res * 100} cm resolution")
    elif  target_res <= 0.0099:
        print(f"Resampling input raster to {target_res * 1000} mm resolution")
    else:
        print(f"Resampling input raster to {target_res} m resolution")

    orig_res=raster.rio.resolution() # Get original resolution of raster
    orig_res_x = orig_res[0] # resolution in x / lon direction
    orig_res_y = abs(orig_res[1]) # resolution in y / lat direction

    orig_width = raster.rio.width # original width
    orig_height = raster.rio.height # original height


    rescale_x=target_res / orig_res_x # X rescale factor
    rescale_y=target_res / orig_res_y # Y rescale factor

    target_width= round(orig_width / rescale_x) # Calculate Target Width
    target_height=round(orig_height / rescale_y) # Calculate Target Height

    # Reample input raster to the new dimensions
    ## Bilinear resampling set as default resampling method
    resampled = raster.rio.reproject(
        raster.rio.crs,
        shape=(target_height, target_width),
        resampling=resampling,
    )

    return resampled



def clip_raster(raster_path, gdf,crs, bbox, out_path, nodata):
    ''' Clip raster from inputted file path to a bounding box

    Args:
        raster_path (str): File path of raster to clip
        gdf (GeoDataFrame): Polygon to clip raster
        bbox (shapely.geometry.Polygon): Polygon of bounding box for clipping raster
        out_path (str): File path of clipped raster
        nodata (int or float): Fill value or No Data value

    Returns:
        None
    '''
  
    # Read Band Tile raster and clip with rasterio.mask
    with r.open(raster_path) as src:
        # clipped_path=SENT2_CLIPPED_DIR + "/" + band_tile_fp.replace(".tif","_clipped.tif").split("/")[-1] # Configure output path of clipped raster (per band)

        if gdf is not None:

            gdf=gdf.to_crs(crs)
            bbox = list(gdf.geometry.values)

        else:
            bbox=[bbox]
        out_meta=src.meta.copy() # copy original metadata
        out_image, out_transform = r.mask.mask(src, bbox, crop=True, nodata=nodata, all_touched=False) # Clip raster


        print(f"Clipped {raster_path.split('/')[-1]}")

        #Update raster metadata with new dimensions and transform
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        'nodata' : nodata
                        })
    
    # Write clipped raster as GeoTIFF
    with r.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Exported clipped raster for {raster_path.split('/')[-1]}")


def rasterize_vector(gdf, ref, attribute=None,out_path=None, crs=None, fill_value=0):
    '''

    Args:
        gdf (GeoDataFrame): Vector polygon GeoDataFrame to rasterize
        src (DataArray): Reference raster to match crs, dimensions and transform
        attribute (str): Column from gdf to burn into output raster, must be an integer type
        out_path (str): Output file path for the raster
        crs: Target coordinate reference system
        fill_value (int): Fill or No Data Value
        
    Returns:
        None
    '''
    # Load vector data using geopandas

  
    try:
        ref=ref.sel(band=1)
    except:
        pass

    # Define raster dimensions (resolution)
    width = ref.rio.shape[1]
    height = ref.rio.shape[0]

    # Create a transform (affine) matrix
    transform = ref.rio.transform()

    # Optionally reproject vector CRS if needed
    if crs:
        gdf = gdf.to_crs(crs)
        ref=ref.rio.reproject(dst_crs=crs)

    # Prepare shapes for rasterization (geometry, value)
    if attribute:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = ((geom, 1) for geom in gdf.geometry)  # default value is 1

    # Rasterize shapes into a numpy array
    raster = r.features.rasterize(
        shapes=shapes, 
        out_shape=(height, width), 
        transform=transform, 
        fill=fill_value)

    if out_path is not None:
        # Save raster to file
        with r.open(
            out_path, 'w', 
            driver='GTiff', 
            height=height, 
            width=width, 
            count=1, 
            dtype=raster.dtype, 
            crs=gdf.crs, 
            transform=transform
        ) as dst:
            dst.write(raster, 1)
    else:
        raster=xr.DataArray(raster, coords=ref.coords)
        raster=raster.rio.write_crs(ref.rio.crs)
        return raster
    





def northing(y):
    ''' Get northing based on latitude

    Args:
        y (float): Latitude coordinate in WGS84 / EPSG:4326 projection
       

    Returns:
        str: Northing value
    '''
    ns = "N" if y >= 0 else "S"
    y = 5 * round(abs(int(y)) / 5)
    return y, ns

def easting(x):
    ''' Get easting based on longitude

    Args:
        x (float): longitude coordinate in WGS84 / EPSG:4326 projection
       

    Returns:
        str: Easting Value
    '''
    ew = "E" if x >= 0 else "W"
    x = 5 * round(abs(int(x)) / 5)
    return x,ew


def get_overlapping_tiles(bbox):
    '''Get overlapping tile names baased on input bounding box

    Args:
        bbox (list): bounding box list ordered x1, y1, x2, y2
       

    Returns:
        list: Names of all tiles that overlap the given bounding box
    '''

    x1,y1,x2,y2 = bbox
    e1,ew1=easting(x1)
    e2,ew2=easting(x2)
    n1,ns1=northing(y1)
    n2,ns2=northing(y2)

    tile_size=5
   
    if ew1 == "W":
        e_list=[f"{x}{ew1}" for x in range(e2 - tile_size, e1 + tile_size, 5)]
    elif ew1 == "E":
        e_list=[f"{x}{ew1}" for x in range(e1 - tile_size, e2 + tile_size, 5)]

    if ns1 == "N":
        n_list = [f"{y}{ns1}" for y in range(n1 - tile_size, n2 + tile_size, 5)]
    elif ns1 == "S":
        n_list = [f"{y}{ns1}" for y in range(n2 - tile_size, n1 + tile_size, 5)]

    ee,yy = np.meshgrid(e_list,n_list)
    coords=np.vstack([yy.ravel(), ee.ravel()]).T

    return ["_".join(ne) for ne in coords]



    
def generate_raster(bbox,crs, resolution):
    '''Generate at empty raster at the desired resolution within the input bounding box

    Args:
        bbox (list): bounding box list ordered x1, y1, x2, y2
        crs: coordinate reference system fo generated raster EPSG number or pyproj.CRS object
        bbox (int): Desired resolution of generated raster in meters
       

    Returns:
        xr.DataArray: Empty raster at the desired resolution within the input bounding box
    '''

    x1,y1,x2,y2=bbox

    crs = pycrs.from_epsg(crs)  # Replace with your CRS or .from_user_input(...)
    # Check the unit
    unit = crs.axis_info[0].unit_name

    if unit == 'degree':
        y_res, x_res = meter_to_deg(resolution, y1)
    elif unit == 'metre' or unit == 'meter':
        y_res, x_res = (resolution, resolution)

    xx=np.arange(x1,x2, x_res)
    yy=np.arange(y1,y2,y_res)
    xx_grid, yy_grid=np.meshgrid(xx,yy)
    # shape=
    fill=np.zeros_like(xx_grid)
    return xr.DataArray(
        data=fill,
        dims=["y","x"],
        coords=dict(
            y= yy,
            x = xx,
        
        ), 
    ).rio.write_crs(crs,inplace=True)

def jeffries_matuista_distance(class1, class2):
    """
    Compute Jeffries-Matusita (JM) distance using NumPy.
    
    Parameters:
    - class1: np.ndarray of shape (n_samples, n_bands)
    - class2: np.ndarray of shape (n_samples, n_bands)
    
    Returns:
    - JM distance (float)
    """
    # Mean vectors
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    
    # Covariance matrices
    s1 = np.cov(class1, rowvar=False)
    s2 = np.cov(class2, rowvar=False)
    
    # Average covariance matrix
    s12 = 0.5 * (s1 + s2)
    
    # Add small identity matrix for numerical stability
    epsilon = 1e-6
    s12 += np.eye(s12.shape[0]) * epsilon
    s1 += np.eye(s1.shape[0]) * epsilon
    s2 += np.eye(s2.shape[0]) * epsilon

    # Difference of means
    diff = (m1 - m2).reshape(-1, 1)
    
    # First term of Bhattacharyya distance
    B1 = 0.125 * (diff.T @ np.linalg.inv(s12) @ diff).item()
    
    # Second term
    det_s1 = np.linalg.det(s1)
    det_s2 = np.linalg.det(s2)
    det_s12 = np.linalg.det(s12)
    
    B2 = 0.5 * np.log(det_s12 / np.sqrt(det_s1 * det_s2))
    
    # Bhattacharyya distance
    B = B1 + B2
    
    # Jeffries-Matusita distance
    JM = 2 * (1 - np.exp(-B))
    
    return JM


def band_stats(zones,raster,stats=['min', 'max', 'mean', 'median', 'majority']):
    '''Compute band statsitics from a Geopandas GeoDataFrame and an xarray DataArray


    Args:
        zones (np.array): Classified zones for raster statistics
        raster (xr.Datarray): Stack of features for extracting statistics
        stats (list): Desired statistics
       

    Returns:
        DataFrame: Statistics of each class from each band
    '''
  
    affine=raster.rio.transform()
    stats_df_list=list()
    for band in raster.band:
  
        band=band.values
        array=raster.sel(band=band).values
        

        stats_dict = zonal_stats(zones, array, affine=affine, stats=stats)

        stats_df=pd.DataFrame(stats_dict, index=zones.index)
        stats_df["band"] = band
        stats_df= stats_df.reset_index()
        stats_df_list.append(stats_df)
        
    
    return pd.concat(stats_df_list)


def prepare_dataset(X, y, X_names=None):

    ''' 
    X: np.array
        Stacked predictor features with shape (bands, width, height)

    y: np.array
        Target features (classes) with shape (1, width, height)

    X_names: list
        List of feature names of matching legnth of all X features

    '''
    # combine arrays and mask them with training areas
    train_data=np.append(X,y, axis = 0)
    mask = (y > 0).values[0] 
    masked=np.array([arr[mask] for arr in train_data]) # Mask out null values 

    # Prepare column names for train_df
    cols=X_names.copy()
    cols.append('y')

    # Make DataFrame of for filtering out remaining nan values
    df=pd.DataFrame(masked).T.dropna()
    df.columns=cols
    clean_df=df[~df.isna()]

    # Reshape arrays for model fitting (width*height, bands)
    X_df=clean_df[X_names].values.reshape(-1,X.shape[0])
    y_df = clean_df['y'].values.reshape(-1)

    return X_df, y_df


def dataset_stats(X,y, label_dict):
    '''Get statistics of machine learning dataset

    Args:
        X (np.array): 2D Array shaped (total_pixels, n_features)
        y (np.array): 1D Array shaped (total_pixels)
        label_dict (dict): mapping between y values to label names
       

    Returns:
        dict: Dataset statistics in a dictionary object
    '''
    
    stats=dict()

    stats['samples']=X.shape[0]
    stats['features']=X.shape[-1]
    classes,class_counts = np.int64(np.unique(y, return_counts=True))
    stats['n_classes']=len(classes)

    if label_dict:
        classes=[label_dict[x] for x in classes]
    stats['class_counts'] =dict(zip(classes,class_counts))
    
    return stats


def dataset_summary(train_stats,test_stats):
    '''Print Dataset Summary

    Args:
        train_stats (dict): Statistics of training dataset, outputted from dataset_stats()
        test_stats (dict): Statistics of testing dataset, outputted from dataset_stats()
      
    '''
    
    table=PrettyTable()
    table.field_names=["Datset","Samples", "Features", "Classes"]
    table.add_rows([
        ['Train',train_stats['samples'], train_stats['features'], train_stats['n_classes']],
        ['Test', test_stats['samples'], test_stats['features'], test_stats['n_classes']]
    ])

   
    print("============== DATASET SUMMARY ==============")
    print(table)

    class_table=PrettyTable()
    class_table.field_names=["Class", "Train", "Test"]
    table_vals=[list(x["class_counts"].values()) for x in [train_stats,test_stats]]
    classes=[list(test_stats['class_counts'].keys())]
    # table_vals= np.append(classes, table_vals)
    classes.extend(table_vals)
    table_vals=list(np.array(classes).T)
    class_table.add_rows(table_vals)
    print("")
    print("============-==== CLASS SUMMARY =================")
    print(class_table)



def meter_to_deg(meters, latitude):
    """
    Convert distance in meters to approximate decimal degrees at a given latitude.
    
    Args:
        meters (float): Cell size in meters.
        latitude (float): Latitude in decimal degrees where the conversion is needed.
        
    Returns:
        tuple: (degrees_lat, degrees_lon)
    """
    # Earth radius in meters (WGS84 approximation)
    earth_radius = 6378137

    # Degrees latitude per meter (fairly constant)
    degrees_lat = meters / 111320  # ~111.32 km per degree

    # Degrees longitude per meter (varies by latitude)
    lat_rad = math.radians(latitude)
    meters_per_deg_lon = (math.pi / 180) * earth_radius * math.cos(lat_rad)
    degrees_lon = meters / meters_per_deg_lon

    return degrees_lat, degrees_lon


