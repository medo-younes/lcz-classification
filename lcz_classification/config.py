
import pandas as pd
import arrow
import geopandas as gpd
from shapely.geometry import box
import os


## FUNCTIONS ## 
def setup_dir(root_dir, parent, raw, prc):

    parent_path=f"{root_dir}/{parent}"
    study_area_path=f"{root_dir}/{parent}/study_area"
    raw_path = f'{parent_path}/raw'
    prc_path = f'{parent_path}/processed'

    raw_paths = [f"{raw_path}/{p}" for p in raw]
    prc_paths = [f"{prc_path}/{p}" for p in prc]
    
    paths=[parent_path, study_area_path, raw_path, prc_path]
    paths.extend(raw_paths)
    paths.extend(prc_paths)
    

    if os.path.exists(parent_path):
        print(f"- Parent directory for {parent.title()} already exists, retrievine folder paths...")
    
    else:
    
        for path in paths:
            os.mkdir(path)
        print(f"- Data directories for {parent.title()} created.")

    print(f"- Folder paths for {parent.title()} retrieved.")
    return parent_path, study_area_path, raw_paths, prc_paths

###
## USER INPUT ##
# ===============================================================================================================================================================

CRS=4326
# Set up Geographic Bounds
CITY = "Toronto" # Select City Name
COUNTRY = "Canada" # Select Country Name


# Setup Date Range
TARGET_DATE = '2020-06-22'
START_DATE = '2020-06-01' # Start date in YYYY-MM-DD
END_DATE ='2020-06-30' # End date in YYYY-MM-DD
DATE_RANGE=[arrow.get(START_DATE, 'YYYY-MM-DD'), arrow.get(END_DATE, 'YYYY-MM-DD')] 

## SENTINEL-2 MEtaData
S2_SCALE=0.0001
S2_TARGET_BANDS=["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"] #Sentinel-2, bands from B02 to B07, B8A, B11, and B12 (provided at 20 m spatial resolution by Copernicus) are exploited.

## WORFLOW PARAMETERS
CELL_RESOLUTION = 30 # desired resolution in meters
TILE_DIMS = (1,2)

## DIRECTORY PATHS ##
# ===============================================================================================================================================================

# Setup Primary Raw and Procesed data directories
META=f'../data/meta'
RAW= ['sentinel2','lcz','alos_dsm_30m','canopy_height','building_height','impervious_surface']
PRC=['sentinel2', 'sentinel2/merged', 'sentinel2/clipped', 'sentinel2/resampled','training_data','canopy_height','building_height','svf','impervious_surface','classification']

DATA, STUDY_AREA, RAW_PATHS, PRC_PATHS = setup_dir(
                                    root_dir="../data",
                                    parent=CITY.lower(), 
                                    raw=RAW, 
                                    prc=PRC
                                 )

RAW_DIR = f'../data/{CITY.lower()}/raw'
PRC_DIR = f'../data/{CITY.lower()}/processed'

# Get all paths from setup_dir()
S2_RAW, LCZ_RAW, DSM_RAW, CH_RAW, BH_RAW, IS_RAW = RAW_PATHS # Raw Directories
S2_PRC, S2_MERGED, S2_CLIPPED, S2_RESAMPLED, TRAIN_PRC, CH_PRC, BH_PRC, SF_PRC, IS_PRC, CL_PRC = PRC_PATHS # Processed Directories

IS_RAW=f"../data/impervious_surface"

# Figures Directory
FIGURES_DIR="../reports/figures/"

## FILE PATHS ##
# ===============================================================================================================================================================

# METADATA AND LEGENDS
# STUDY_AREA_FP=f"{STUDY_AREA}/{CITY.lower()}.geojson" # Must have created the study area file from  01_Data_Aquisition.ipynb
STUDY_AREA_FP=f"{STUDY_AREA}/{CITY.lower()}.geojson"
S2_METADATA_FP=f"{META}/sentinel_bands.csv"
S2_TILES_FP = f"{META}/sentinel_2_tiles.geojson" # 
LCZ_LEGEND_FP = f"{META}/lcz_legend.csv" # LCZ Legend - mapping for classes and colors

## PROCESSED FILE PATHS ##
# ===============================================================================================================================================================

# DSM / SKY VIEW FACTOR
DSM_PATH = f"{SF_PRC}/alos_dsm_{CELL_RESOLUTION}m.tif"
SVF_FP=f'{SF_PRC}/svf_{CELL_RESOLUTION}m.tif'

# IMPERVIOUS SURFACE
IS_MERGED_FP = f"{IS_PRC}/gisa_10m.tif"

# BUILDING HEIGHT
BH_VECTOR_FP = f"{BH_PRC}/building_height.parquet"
BH_RASTER_FP=f"{BH_PRC}/building_height_5m.tif"

# LOCAL CLIMATE ZONE TRAINNG AREAS
LCZ_FP=f'{LCZ_RAW}/training_areas_v2.kml'
# LCZ_FP=f"{TRAIN_PRC}/lcz_{CITY.lower()}.geojson"


# MODEL TRAIN / TEST DATA
TRAIN_FP=f"{TRAIN_PRC}/train_{CELL_RESOLUTION}m.tif"
TEST_FP=f"{TRAIN_PRC}/test_{CELL_RESOLUTION}m.tif"


# MODEL TRAINING FEATURES
S2_FP=f"{S2_RESAMPLED}/s2_{CELL_RESOLUTION}m.tif" # 
BH_FP=f"{TRAIN_PRC}/building_height_{CELL_RESOLUTION}m.tif" # Building Height
CH_FP=f"{TRAIN_PRC}/canopy_height_{CELL_RESOLUTION}m.tif" # Tree Canopy Height
IS_FP=f"{TRAIN_PRC}/impervious_surface_fraction_{CELL_RESOLUTION}m.tif" # Impervious Surface Fraction
BS_FP=f"{TRAIN_PRC}/building_surface_fraction_{CELL_RESOLUTION}m.tif" # Building Surface Fraction
SF_FP=f"{TRAIN_PRC}/svf_{CELL_RESOLUTION}m.tif" # Building Surface Fraction

UCPS=[BH_FP, CH_FP, IS_FP, BS_FP, SF_FP] # Combined list of all UCP file paths 

# CLASSIFIED IMAGE FILE PATH
CLASSIFIED_FP=f'{CL_PRC}/classified_{CELL_RESOLUTION}m.tif'
CLASSIFIED_CLIPPED_FP=f'{CL_PRC}/classified_{CELL_RESOLUTION}m_clip.tif'


# LST DATA
LST_RAW=f'{RAW_DIR}/lst'
LST_PRC=f'{PRC_DIR}/lst'
LST_FP=f'{LST_PRC}/nightime_lst_{CELL_RESOLUTION}_202307.tif'