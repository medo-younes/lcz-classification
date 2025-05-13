# Urban Heat Island Analysis with Automated Local Climate Zone Classification: A Cairo Case Study
Automated classification of Local Climate Zones (LCZs) using Random Forest trained on Sentinel-2 L2A imagery and GIS-derived Urban Canopy Parameters for intra-urban urban heat island analysis in Cairo, Egypt.

<!-- <p align="center">
  <img src="reports/figures/lcz_s2_map.png" alt="Classified LCZ Map" width="500"/>
</p>
 -->

### 1. Local Climate Zones

The Urban Heat Island (UHI) effect is a phenomenon whereby air temperatures in urban environments are significantly heightened when compared to rural areas. Such an effect is largely due to the thermal properties of urban structures, which tend to absorb heat during the day and release it during the night. Traditional studies typically measured UHI using dichotomous urban / rural classification when comparing temperature trends. Such an approach fails to capture the diverse nature of urban areas, whereby building height, compactness and vegetation cover can vary substantially over space. Developed Stewart and Oke (2012), the Local Climate Zone (LCZ) classification scheme aims to characterise The scheme comprises 17 zones based mainly on properties of surface structure (e.g., building and tree height & density) and surface cover (pervious vs. impervious). Each zone is local in scale, meaning it represents horizontal distances of 100s of metres to several kilometres.  The scheme is a logical starting point for WUDAPT’s aim to gather consistent information across cities globally. To learn more about the Local Climate Zone framework, you can refer to the [WUDAPT Webpage](https://www.wudapt.org/lcz/). A useful resource to better undetstand LCZ can be found in this [illustration by Demuzere et al (2020)](https://www.wudapt.org/wp-content/uploads/2021/03/LCZ_Typology_Demuzere2020.pdf).

<!-- ### 2. A Toronto Case Study

This project explores the application of LCZs for understanding the UHI in the city of Toronto, Ontario. A study in 2021 found that Toronto experienced an annual average daytime UHI intensity of 4.3 C (Duan et al. 2024).

Primary Objectives:
1. Train a Random Forest classifier on Sentinel-2 Imagery and GIS-derived Urban Canopy Parameters to predict LCZ classes for the city of Toronto
2. Analyze the Urban Heat Island Effect in the Toronto using classified LCZs  -->


### 3. Datasets

The table below outlines the various datasets employed for conducting LCZ classification in Cairo. Except for the Canadian buildings dataset, all datasets are available with global coverage.

|Name      |Spatial Resolution| Reference Date|Source | 
|------------|------------|------------|------------|
|Sentinel-L2A   | 10 / 20 m  | 2023 | [Google Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#description)|
|ALOS DSM: Global (30m) v3.2  | 30 m  | 2006 | [Google Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V3_2)|
|Open Buildings 2.5D Temporal Dataser  |  4 m | 2023| [Google Research](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1)|
|ETH Global Sentinel-2 (10m) Canopy Height  | 30 m  |2020 | [Google Earth Engine Catalog](https://gee-community-catalog.org/projects/canopy/)|
|GISA-10m Impervious Surface Area | 10 m  |2016 | [Huang et al (2021)](https://zenodo.org/records/5791855)|


<!-- #### Local Climate Zone Training Areas

A total of 317 training area polygons representing 12 LCZ classes were employed. The polygons were widely distributed around Toronto, enabling effective training of the Random Forest classifier. -->

<!-- <img src="reports/figures/lcz_training_areas.png" alt="drawing" width="500"/> -->

#### Sentinel-2 L2A Imagery

As per Vavassori et al. (2024), bands from B02 to B07, B8A, B11, and B12 were utilized from the harmonized Sentinel-2 L2A dataset. Below is an RGB composite of the Sentinel-2 imagery collected on 15th of May, 2023.

#### Spectral Signature of LCZ Classes 

<img src="reports/figures/s2_spectral_signature.png" alt="drawing" width="500"/>


Spectral Seperability Between Classes using Jeffries-Matuista Distance

<img src="reports/figures/s2_jm.png" alt="drawing" width="500"/>


#### Urban Canopy Parameters

Following the steps outlined by Vavassori et al. (2024), Urban Canopy Parameters for Toronto were derived.

|UCP      | Source Dataset| 
|------------|------------|
|Building Height  | [Google Research](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1)| 
|Tree Canopy Height| [ETH Global Sentinel-2 (10m) Canopy Height](https://gee-community-catalog.org/projects/canopy/) | 
|Sky View Factor  |[ALOS DSM: Global (30m) v3.2 ](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V3_2) |
|Impervious Surface Fraction | [GISA-10m Impervious Surface Area](https://zenodo.org/records/5791855)|
|Building Surface Fraction| [Government of Canada](https://open.canada.ca/data/en/dataset/7a5cda52-c7df-427f-9ced-26f19a8a64d6)|


<!-- <img src="reports/figures/UCPs.png" alt="drawing" width="500"/> -->

### 3. LCZ Classification with Random Forest

A Random Forest Classifier was trained on Sentinel-2 imagery and GIS-derived Urban Canopy Parameters to predict LCZ classes, data processing, data exploration and model assessment follows the methodology from Vavasorri et al. (2024). The trained model attained an overall testing accuracy of 90%, the classification report, confusion matrix and feature importances are given below.

### Accuracy Metrics

|Metric      |Result|
|------------|------|
|Accuracy    |0.92  |
|Macro Avg   |0.76 |
|Weighted Avg |0.92  |

### Classification Report

|Class         |Precision|Recall|F1-score|Support|
|------------------|---------|------|--------|-------|
|Compact High-Rise |0.86     |0.8   |0.83    |664    |
|Open High-Rise    |0.67     |0.7   |0.69    |1166   |
|Open Mid-Rise     |0.73     |0.44  |0.55    |471    |
|Open Low-Rise     |0.87     |0.91  |0.89    |2741   |
|Large low-rise    |0.86     |0.89  |0.87    |1596   |
|Sparsely built    |0.72     |0.65  |0.68    |871    |
|Dense trees       |0.9      |0.96  |0.93    |593    |
|Scattered trees   |0.88     |0.92  |0.90     |451    |
|Low plants        |0.96     |0.96  |0.96    |2424   |
|Bare rock or paved|0.72     |0.91  |0.80     |380    |
|Bare soil or sand |0.95     |0.91  |0.93    |553    |
|Water             |1.0      |1.0   |1.0     |1570   |



#### Confusion Matrix

<img src="reports/figures/s2_cm.png" alt="drawing" width="500"/>

#### Feature Importances
<img src="reports/figures/s2_fi.png" alt="drawing" width="500"/>

### 4. Urban Heat Island Analysis

Thermal differences between urban and rural sites are most pronounced during the night, therefore this study analyzes nightime temperature data in the context of LCZs. [MOD11A1.061 Terra Land Surface Temperature and Emissivity Daily Global Dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1)  provides nightime Land Surface Temperature (LST) at 1km spatial resolution, which is available via the LST_Night_1km band. Toronto's mean nighttime LST data was retrieved for July 2023 using the Google Earth Engine Python API. 

By calculating temperature statistics of each classified LCZ, we can observe distinct thermal behaviours of highly localized urban and rural sites in the City of Toronto. On average, built-up LCZs exhibited higher nightime temperatures than non-built types by at least 1°C. Compact High-rise had the highest mean nighttime LST (21.4°C) amongst all LCZs, while Low Plants had the lowest (18.1°C). When compared to mean nighttime LST across the whole city (20.1°C), Compact High-rise was approximately 1.3°C higher and Low Plants was nearly 2°C lower.



#### LCZ Nightime Temperature Statistics (July 2023)

|Local Climate Zone|Mean |Min  |Max  |Median|Standard Deviaton|Range|
|------------------|-----|-----|-----|------|----|-----|
|Compact High-Rise |21.4 |16.0 |24.0 |21.4  |0.8 |7.9  |
|Bare rock or paved|21.1 |16.3 |24.0 |21.0  |1.2 |7.6  |
|Large low-rise    |21.0 |15.3 |24.0 |21.1  |1.3 |8.6  |
|Open High-Rise    |21.0 |15.3 |24.0 |21.2  |1.5 |8.6  |
|Open Low-Rise     |21.0 |15.3 |24.0 |21.2  |1.4 |8.6  |
|Open Mid-Rise     |20.7 |15.3 |24.0 |21.1  |1.7 |8.6  |
|Scattered trees   |19.7 |15.3 |23.6 |20.0  |2.0 |8.3  |
|Water             |19.5 |16.2 |22.9 |19.5  |0.4 |6.7  |
|Sparsely built    |19.5 |15.3 |23.6 |19.7  |2.1 |8.3  |
|Dense trees       |19.0 |15.3 |23.6 |18.9  |2.1 |8.3  |
|Bare soil or sand |19.0 |16.2 |24.0 |18.8  |1.7 |7.8  |
|Low plants        |18.1 |15.3 |24.0 |17.1  |1.9 |8.6  |



#### Thermal Trends Amongt LCZs
<img src="reports/figures/temp_barplots.png" alt="drawing" width="800"/>

#### Pairwise Temperature Differences between LCZs
<img src="reports/figures/temp_pairwise.png" alt="drawing" width="500"/>


### References


Alberto Vavassori, Daniele Oxoli, Giovanna Venuti, Maria Antonia Brovelli, Mario Siciliani de Cumis, Patrizia Sacco, Deodato Tapete, A combined Remote Sensing and GIS-based method for Local Climate Zone mapping using PRISMA and Sentinel-2 imagery, International Journal of Applied Earth Observation and Geoinformation, Volume 131, 2024, 103944, ISSN 1569-8432, https://doi.org/10.1016/j.jag.2024 103944.

Ching, J., Mills, G., Bechtel, B., See, L., Feddema, J., Wang, X., Ren, C., Brousse, O., Martilli, A., Neophytou, M., Mouzourides, P., Stewart, I., Hanna, A., Ng, E., Foley, M., Alexander, P., Aliaga, D., Niyogi, D., Shreevastava, A., Bhalachandran, P., Masson, V., Hidalgo, J., Fung, J., Andrade, M., Baklanov, A., Dai, W., Milcinski, G., Demuzere, M., Brunsell, N., Pesaresi, M., Miao, S., Mu, Q., Chen, F., Theeuwes, N., 2018. WUDAPT: An Urban Weather, Climate, and Environmental Modeling Infrastructure for the Anthropocene. Bull. Amer. Meteor. Soc. 99, 1907–1924. https://doi.org/10.1175/BAMS-D-16-0236.1

Demuzere M, Hankey S, Mills G, Zhang W, Lu T, Bechtel B. Combining expert and crowd-sourced training data to map urban form and functions for the continental US. Sci Data. 2020;7(1):264. doi:10.1038/s41597-020-00605-z.

Demuzere, M., Kittner, J., Bechtel, B. (2021). LCZ Generator: a web application to create Local Climate Zone maps. Frontiers in Environmental Science 9:637455. https://doi.org/10.3389/fenvs.2021.637455

Duan, Yuwei and Agrawal, Sandeep and Sanchez-Azofeifa, Arturo and Welegedara, Nilusha, Urban Heat Island Effect in Canada: Insights from Five Major Cities. Available at SSRN: https://ssrn.com/abstract=4965331 or http://dx.doi.org/10.2139/ssrn.4965331

Stewart ID, Oke TR. Local Climate Zones for Urban Temperature Studies. Bull Am Meteorol Soc. 2012;93(12):1879-1900. doi:10.1175/BAMS-D-11-00019.1

