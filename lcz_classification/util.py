

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
fiona.supported_drivers['KML'] = 'r'  # Explicitly enable KML read support

def kml_to_gdf(kml_path):
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
    return scaler.fit_transform(arr.reshape(-1,1)).reshape(arr.shape)


def read_lcz_legend(file_path):
    lcz_legend=pd.read_csv(file_path)
    lcz_legend["hex"]=lcz_legend.apply(lambda row: '#%02x%02x%02x' % (row.r,row.g,row.b) , axis=1)
    lcz_legend['rgb'] = lcz_legend.apply(lambda x: (x["r"],  x["g"], x["b"]), axis = 1)
    color_dict=lcz_legend[["name", "hex"]].set_index("name").hex.to_dict()

    return lcz_legend, color_dict

def tiles_from_bbox(bbox, crs=4236, tile_dims=(4,4)):
    """
    Generates polygon tiles of 
    
    Parameters:
    -----------
    bbox : list
        Bounding box coordinates in format [x1,y1,x2,y2]
    tile_dims : tuple
        How many tiles to sub-section the bounding box with the dimensions (width, length)
        With tile_dims == (4,4) the function will return 16 (4 x 4) tile polygons

    Returns:
    --------
    GeoDataFrame
        Geopandas GeoDataFrame of generated tiles as Shapely Polygons
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
    return tiles


def merge_rasters(raster_paths, out_path):
    rasters= [r.open(path) for path in raster_paths]
    array, transform = merge(rasters)

    # Use metadata from one of the source files and update it
    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "height": array.shape[1],
        "width": array.shape[2],
        "transform": transform
    })

    # Write the merged output
    with r.open(out_path, "w", **out_meta) as dest:
        dest.write(array)
    


def get_target_shape(raster,target_res):

    '''

    Resample an xarray DataArray to a targett resolution
    

    Parameters
    ----------
    raster: xr.DataArray
    Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
    
    target_res : int
    Cell resolution in METERS that you want to resample the raster to

    Outputs
    ---------

    rescaled: xr.DataArray
    Raster rescaled to the target resolution, keeping the original projection of the input raster
    
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

    '''

    Resample an xarray DataArray to a targett resolution
    

    Parameters
    ----------
    raster: xr.DataArray
    Raster layer you would like to resample. CRS MUST BE IN METERS, WILL NOT WORK WITH EPSG:4326
    
    target_res : int
    Cell resolution in METERS that you want to resample the raster to

    Outputs
    ---------

    rescaled: xr.DataArray
    Raster rescaled to the target resolution, keeping the original projection of the input raster
    
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



def clip_raster(raster_path, gdf, bbox, out_path):
    
    '''

    Clip raster from inputted file path to a bounding box
    

    Parameters
    ----------
    raster_path: str
        File path of raster to clip
    
    bbox : Polygon
        Shapely Polygon of bounding box for clipping raster

    out_path: str
        Ouptut file path of clipped raster

    Outputs
    ---------

    If out_path is not provided, returns the clipped raster
    '''

    # Read Band Tile raster and clip with rasterio.mask
    with r.open(raster_path) as src:
        # clipped_path=SENT2_CLIPPED_DIR + "/" + band_tile_fp.replace(".tif","_clipped.tif").split("/")[-1] # Configure output path of clipped raster (per band)

        if gdf is not None:

            gdf=gdf.to_crs(src.crs)
            bbox = list(gdf.geometry.values)

        else:
            bbox=[bbox]
        out_meta=src.meta.copy() # copy original metadata
        out_image, out_transform = r.mask.mask(src, bbox, crop=True, all_touched=True) # Clip raster
        print(f"Clipped {raster_path.split('/')[-1]}")

        #Update raster metadata with new dimensions and transform
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
    
    # Write clipped raster as GeoTIFF
    with r.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Exported clipped raster for {raster_path.split('/')[-1]}")


def rasterize_vector(gdf, ref, attribute=None,out_path=None, crs=None, fill_value=0):
    '''

        gdf : GeoDataFrame
            Vector polygon GeoDataFrame to rasterize
        
        src : DataArray
            Reference raster to match crs, dimensions and transform

        attribute : str
            Column from gdf to burn into output raster, must be an integer type

        out_path : str
            Output file path for the raster
        
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
    



import math


def northing(y):
    ns = "N" if y >= 0 else "S"
    y = 5 * round(abs(int(y)) / 5)
    return y, ns
def easting(x):
    ew = "E" if x >= 0 else "W"
    x = 5 * round(abs(int(x)) / 5)
    return x,ew


def get_overlapping_tiles(bounds):

    x1,y1,x2,y2 = bounds
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
    """
    Compute band statsitics from a Geopandas GeoDataFrame and an xarray DataArray
    
    Parameters:
    - class1: np.ndarray of shape (n_samples, n_bands)
    - class2: np.ndarray of shape (n_samples, n_bands)
    
    Returns:
    - JM distance (float)
    """
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
    stats=dict()

    stats['samples']=X.shape[0]
    stats['features']=X.shape[-1]
    classes,class_counts = np.int16(np.unique(y, return_counts=True))
    stats['n_classes']=len(classes)

    if label_dict:
        classes=[label_dict[x] for x in classes]
    stats['class_counts'] =dict(zip(classes,class_counts))
    
    return stats


def dataset_summary(train_stats,test_stats):
    
    table=PrettyTable()
    table.field_names=["Datset","Samples", "Features", "Classes"]
    table.add_rows([
        ['Train',train_stats['samples'], train_stats['features'], train_stats['n_classes']],
        ['Test', test_stats['samples'], test_stats['features'], test_stats['n_classes']]
    ])

    # print(f"Training Features: {X_train.shape[1]}")
    # print(f"Training Samples: {X_train.shape[0]}")
    # print(f"Validation Samples: {X_val.shape[0]}")
    # print(f"Testing Samples: {X_test.shape[0]}")
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
    
    Parameters:
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

# def open_layer(layer_path):
#   with rio.open(layer_path) as dataset:
#     layer = dataset.read().squeeze()

#     layer[layer < 0] = 0
#     #layer = np.nan_to_num(layer, nan=0)
#     min_val = np.min(layer)
#     max_val = np.max(layer)

#     if max_val > 1:
#       normalized_array = (layer - min_val) / (max_val - min_val)
#       print(f"{layer_path} shape: {layer.shape} ---> Max value: {np.max(normalized_array):.2f} | Min value: {np.min(normalized_array):.2f}")
#       return normalized_array
#     else:
#       print(f"{layer_path} shape: {layer.shape} ---> Max value: {np.max(layer):.2f} | Min value: {np.min(layer):.2f}")
#       return layer

# def check_layers_dimension(imperv, perc_build, svf, canopy_height, buildings, roi, img):

#     # Check dimensions and compute new layers if necessary
#     array_dim = 3 # if they already have dimension 3 don't expand dimensions

#     if imperv.ndim < array_dim:
#         imperv = np.expand_dims(imperv, axis=-1)
#     if perc_build.ndim < array_dim:
#         perc_build = np.expand_dims(perc_build, axis=-1)
#     if svf.ndim < array_dim:
#         svf = np.expand_dims(svf, axis=-1)
#     if canopy_height.ndim < array_dim:
#         canopy_height = np.expand_dims(canopy_height, axis=-1)
#     if buildings.ndim < array_dim:
#         buildings = np.expand_dims(buildings, axis=-1)


#     print("Impervious shape: ", imperv.shape)
#     print("Build percentage shape: ", perc_build.shape)
#     print("SVF shape: ", svf.shape)
#     print("Tree Canopy Height shape: ", canopy_height.shape)
#     print("Building shape: ", buildings.shape)
#     print("ROI shape: ", roi.shape)
#     print("Landsat image: ", img.shape)


#     # Calculate the difference in width between the current shape and the target shape
#     width_diff = np.abs(imperv.shape[0] - roi.shape[0])
#     height_diff = np.abs(imperv.shape[1] - roi.shape[1])
#     print(roi.shape)
#     print(width_diff, height_diff)

#     # Pad the array with zeros along the width
#     roi = np.pad(roi, ((0, 0), (0, width_diff)), mode='constant')
#     roi = np.pad(roi, ((0, 0), (0, height_diff)), mode='constant')
#     print(f"The ROI shape is --> {roi.shape}")

#     return imperv, perc_build, svf, canopy_height, buildings, roi




# def export_classified_map(img, clc, X, selected_image):

#     img = np.nan_to_num(img)

#     # reshape into long 2d array (nrow * ncol, nband) for classification
#     new_shape = (img.shape[0] * img.shape[1], img.shape[2])

#     img_as_array = img[:, :, :X.shape[1]].reshape(new_shape)
#     print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

#     # Now predict for each pixel
#     class_prediction = clc.predict(img_as_array)

#     # Reshape our classification map
#     class_prediction = class_prediction.reshape(img[:, :, 0].shape)

#     # Save the images in GeoTIFF format
#     prisma_image = rio.open(selected_image)
#     kwargs = prisma_image.meta
#     kwargs.update(
#         dtype=rio.float32,
#         nodata = np.nan,
#         count=1)

#     folder_path = "classified_images"

#     # Check if the folder doesn't exist
#     if not os.path.exists(folder_path):
#         # Create the folder
#         os.makedirs(folder_path)
#         print("Folder created successfully.")
#     else:
#         print("Folder already exists.")

#     # Application of the median filter
#     print(f"Application of a median filter of size 3...")
#     # define the size of the median filter window
#     filter_size = 3
#     # apply the median filter to the classified image
#     smoothed_image = median_filter(class_prediction, size=(filter_size, filter_size))

#     # save the classified image with rasterio
#     with rio.open(output_file_path, 'w', **kwargs) as dst:
#         dst.write(smoothed_image, 1)
#         print(f"The smoothed classified file {output_file_path} has been created!")

#     from google.colab import files
#     files.download(output_file_path)

#     return class_prediction, smoothed_image


# def print_accuracy(classified_image, testing, legend):

#     with rio.open(classified_image) as src:
#         mappa = src.read()
#         modified_mappa = mappa.copy()
#         modified_mappa[modified_mappa == 0] = np.nan
#         unique_values_mappa = np.unique(modified_mappa)

#     with rio.open(testing) as src:
#         test = src.read()
#         modified_test = test.astype('float')
#         modified_test[modified_test == 0] = np.nan
#         unique_values_test = np.unique(modified_test)

#     # Retrieve the indices where testing_samples is greater than 0
#     x, y, z = np.where(modified_test>0)

#     # Select the testing samples that are greater than 0
#     test_samples = modified_test[modified_test > 0]
#     # Extract the corresponding classified samples from classified_image using the indices
#     classified_samples = modified_mappa[x, y, z]

#     # Retrieve the indices where classified_samples are greater than 0 and not NaN
#     a = np.where((classified_samples > 0) & (~np.isnan(classified_samples)))

#     # Filter the test_samples and classified_samples based on the indices obtained
#     test_samples = test_samples[a]
#     classified_samples = classified_samples[a]

#     print('Test samples shape:', test_samples.shape)
#     print('Classified samples shape:', classified_samples.shape)

#     # Calculate the accuracy of the best model on the test set
#     accuracy = accuracy_score(test_samples, classified_samples)
#     print(f"OVERALL ACCURACY: {accuracy:.3f}")

#     #accuracy, confusion, report = print_metrics(test_samples, classified_samples)
#     accuracy = accuracy_score(test_samples, classified_samples)
#     confusion = confusion_matrix(test_samples, classified_samples)
#     report = classification_report(test_samples, classified_samples)

#     fig = px.imshow(confusion, text_auto=True)

#     # Update x and y ticks
#     fig.update_xaxes(title_text = 'Classified', tickvals = list(range(len(legend.keys()))), ticktext = [legend[key][0] for key in legend])
#     fig.update_yaxes(title_text = 'Reference', tickvals = list(range(len(legend.keys()))), ticktext = [legend[key][0] for key in legend])

#     # Update heatmap size
#     fig.update_layout(width = 800, height = 600, title = 'Confusion matrix')

#     fig.show()

#     # Convert the report to a pandas df
#     report1 = report.strip().split('\n')
#     report_lists = [line.split() for line in report1]
#     report_df = pd.DataFrame(report_lists[2:13])
#     report_df.columns = ['LCZ', 'precision', 'recall', 'f1-score', 'support']
#     report_df.index = [legend[key][0] for key in legend.keys()]
#     report_df = report_df.iloc[:, 1:]

#     return accuracy, confusion, report, report_df



