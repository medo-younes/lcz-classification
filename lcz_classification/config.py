
import pandas as pd

## USER INPUT ##
# ===============================================================================================================================================================

## WORFLOW PARAMETERS
CELL_RESOLUTION = 30 # desired resolution in meters
#Sentinel-2, bands from B02 to B07, B8A, B11, and B12 (provided at 20 m spatial resolution by Copernicus) are exploited.
SENT2_BANDS=["B02", 
             "B03", 
             "B04", 
             "B05", 
             "B06", 
             "B07", 
             "B8A", 
             "B11", 
             "B12"] 

LCZ_CLASSES=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

## DIRECTORIES

EXTERNAL="../data/external"
STUDY_AREA=f"{EXTERNAL}/study_area"
EE_PROJECT_NAME='ee-geoai-medo'
DSM_DIR = f"{EXTERNAL}/alos_dsm_30m"
BH_DIR=f"../data/external/building_height"
SENT2_DIR =f"{EXTERNAL}/sentinel2"
PROCESSED_DIR = "../data/processed"
SENT2_MERGED_DIR=f"{PROCESSED_DIR}/sentinel2/merged"
SENT2_CLIPPED_DIR=f"{PROCESSED_DIR}/sentinel2/clipped"
SENT2_RESAMPLED_DIR=f"{PROCESSED_DIR}/sentinel2/resampled"
TRAINING_DIR=f"{PROCESSED_DIR}/training_data"
SVF_DIR=f'{PROCESSED_DIR}/svf'

## FILE PATHS
DSM_PATH = f"{DSM_DIR}/alos_dsm_30m.tif"

STUDY_AREA_FP=f"{STUDY_AREA}/study_area.geojson" # Must have created the study area file from  01_Data_Aquisition.ipynb


# LOCAL CLIMATE ZONES
LCZ_TRAINING_VERSION=""
LCZ_TRAINING_FP=f"{PROCESSED_DIR}/training_data/lcz_training_{LCZ_TRAINING_VERSION}.geojson"
LCZ_LEGEND_FP=f"{EXTERNAL}/lcz/lcz_legend.csv"

SVF_FP=f'{SVF_DIR}/svf_{CELL_RESOLUTION}m.tif'

S2_BANDS=pd.read_csv(f"{SENT2_DIR}/sentinel_bands.csv")


# TRAINING DATA FILE PATHS
# lcz_version=LCZ_TRAINING_FP.split("/")[-1].split(".")[0]
LCZ_FP = f"{TRAINING_DIR}/lcz_rasterized_{CELL_RESOLUTION}m.tif"
BH_FP=f"{TRAINING_DIR}/building_height_{CELL_RESOLUTION}m.tif" # Building Height
CH_FP=f"{TRAINING_DIR}/canopy_height_{CELL_RESOLUTION}m.tif" # Tree Canopy Height
IS_FP=f"{TRAINING_DIR}/impervious_surface_fraction_{CELL_RESOLUTION}m.tif" # Impervious Surface Fraction
BS_FP=f"{TRAINING_DIR}/building_surface_fraction_{CELL_RESOLUTION}m.tif" # Building Surface Fraction
SF_FP=f"{TRAINING_DIR}/svf_{CELL_RESOLUTION}m.tif" # Building Surface Fraction

UCPS=[BH_FP, CH_FP, IS_FP, BS_FP, SF_FP]


# 