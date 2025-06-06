import processing 
import os

print(os.listdir())

# Sky View Factor Python Script for SAGA GIS in QGIS Python Console
## Open this script in QGIS >=3.28 , make sure SAGA GIS functions are available via the Processing Toolbox


# SET DIRECTORY PATHS - use absolute path
dsm_dir=r"D:\GeoAI\projects\LCZ_Classification\data\cairo\processed\svf"
out_dir=r"D:\GeoAI\projects\LCZ_Classification\data\cairo\processed\svf"

dsm_file='alos_dsm_30m.tif' # INPUT DSM FILE NAME

# OUTPUT FILE PATHS
svf_file='svf_30m.tif'
vis_file='vis_30m.tif'
smp_file='smp_30m.tif'
trn_file='trn_30m.tif'
vds_file='vds_30m.tif'

# SET SVF PARAMETERS
radius=100
method=1
sectors=16

# RUN SKY VIEW FACTOR SAGA SCRIPT
processing.run("saga:skyviewfactor",{
    'DEM': f'{dsm_dir}/{dsm_file}',
    'RADIUS' : radius,
    'METHOD' : method,
    'NDIRS' : sectors,
    'SVF': f'{out_dir}/{svf_file}',
    'VISIBLE' : f'{out_dir}/{vis_file}',
    'SIMPLE' : f'{out_dir}/{smp_file}',
    'TERRAIN' : f'{out_dir}/{trn_file}',
    'DISTANCE' :f'{out_dir}/{vds_file}',
})

print("Sky View Factor Processing Complete")