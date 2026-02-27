# Fill_pole_Hole_Methods
This repository contains the code for the master thesis, **Estimating sea ice volume inside the Arctic pole hole in the Envisat era**, the thesis is published on [DiVA-portal](https://urn.kb.se/resolve?urn=urn:nbn:se:ltu:diva-116569).

There are three different methods to fill the pole hole, gaps of missing satellite data at the geographical poles, with *sea ice thickness* (SIT) in order to estimate the *sea ice volume* (SIV). The methods were validated against *CryoSat-2* (CS-2) from November 2010 to March 2012, the pole hole is defined from 81.5°N to 88°N. Then the methods are used to estimate the total volume in the Arctic region, starting from 66°N, the pole hole is defined from 81.5°N to 90°N.

**Results**: All plots and .txt files are saved in a respective result folder found in each segment. Estimating_SIV contains an extra folder in the result folder, with additional plots used in the thesis.

**cartoplot.py** and **ll_xy.py** are based on code developed by [Robbie Mallett](https://github.com/robbiemallett/custom_modules.git) and have been modified for this project.

## Table of Contents
1. [Data](#1-data)
2. [Halo](#2-halo)
   - [2.1 Halo_vs_CS2.py](#21-halo_vs_cs2py)
   - [2.2 Halo_EnvPeriod.py](#22-halo_envperiodpy)
3. [Ice Age](#3-ice-age)
   - [3.1 Ice_Age_vs_CS2.py](#31-ice-age_vs_cs2py)
   - [3.2 Ice_Age_EnvPeriod.py](#32-ice-age_envperiodpy)
4. [Neural Network](#4-neural-network)
   - [4.1 format_data.py](#41-format_datapy)
   - [4.2 TrainingData.py](#42-trainingdatapy)
   - [4.3 NN.py](#43-nnpy)
        - [4.3.1 Automated_script.py](#431-automated_scriptpy)
        - [4.3.2 hidden_layers_vs_nodes.py](#432-hidden_layers_vs_nodespy)
        - [4.3.3 Inter_calibration_SSMI_SSMIS.py](#433-inter_calibration_ssmi_ssmispy)
   - [4.4 NN_vs_CS2.py](#44-nn_vs_cs2py)
   - [4.5 NN_EnvPeriod.py](#45-nn_envperiodpy)
5. [Estimating SIV](#5-estimating-siv)
   - [5.1 Plot_estimatedSIV_all_methods.py](#51-plot_estimatedsiv_all_methodspy)
   - [5.2 Envisat_vs_CS2.py](#52-envisat_vs_cs2py)



## 1. Data
The data is in one main folder "Data", then sorted in different nested folders for; each type (eg. Envisat_SatSwath), year (eg. 2010) and month (eg. 11). The downloaded data is for following time periods:
- Cryosat_Monthly: November 2010 – March 2012
- Envisat_Monthly: October 2002 – March 2012
- Envisat_SatSwath: October 2002 – March 2012
    - Months: 01–04, 10–12
- SIC: October 2002 – March 2012
    - Months: 01–04, 10–12
- SIType: October 2002 – March 2012
    - Months: 01–04, 10–12
- TB_SSMI: October 2002 – April 2006
    - Months: 01–04, 10–12
- TB_SSMIS: March 2006, October 2006 – March 2012
    - Months: 01–04, 10–12  

The data can be downloaded from:   
CS-2 SIT data 
[Hendricks, S.; Paul, S.; Rinne, E. (2018): ESA Sea Ice Climate Change Initiative (Sea_Ice_cci): Northern hemisphere sea ice thickness from the CryoSat-2 satellite on a monthly grid (L3C), v2.0. Centre for Environmental Data Analysis. Accessed: 2025-08-14.](https://dx.doi.org/10.5285/ff79d140824f42dd92b204b4f1e9e7c2) 

Envisat monthly grid SIT data 
[Hendricks, S.; Paul, S.; Rinne, E. (2018): ESA Sea Ice Climate Change Initiative (Sea_Ice_cci): Northern hemisphere sea ice thickness from the Envisat satellite on a monthly grid (L3C), v2.0. Centre for Environmental Data Analysis. Accessed: 2025-08-14.](https://dx.doi.org/10.5285/f4c34f4f0f1d4d0da06d771f6972f180)  

Envisat satellite swath SIT data 
[Hendricks, S.; Paul, S.; Rinne, E. (2018): ESA Sea Ice Climate Change Initiative (Sea_Ice_cci): Northern hemisphere sea ice thickness from Envisat on the satellite swath (L2P), v2.0. Centre for Environmental Data Analysis. Accessed: 2025-08-14.](https://dx.doi.org/10.5285/54e2ee0803764b4e84c906da3f16d81b) 

*Sea ice concentration* (SIC) data [OSI SAF Global sea ice concentration climate data record 1978-2020 (v3.0, 2022), OSI-450-a, doi:10.15770/EUM_SAF_OSI_0013. EUMETSAT Ocean and Sea Ice Satellite Application Facility. Data extracted from OSI SAF FTP server: reprocessed/ice/conc/v3p1, (January 2010 - March 2012), (Northern Hemisphere). Accessed 2025-10-13.](https://osi-saf.eumetsat.int/products/osi-450-a) 

SIType data [Copernicus Climate Change Service, Climate Data Store, (2020): Sea ice edge and type daily gridded data from 1978 to present derived from satellite observations. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.29c46d83. Accessed: 2025-10-25.](https://cds.climate.copernicus.eu/datasets/satellite-sea-ice-edge-type?tab=overview) 

*Brightness temperature* (TB) data [Fennig, Karsten; Schröder, Marc; Konrad, Hannes; Hollmann, Rainer (2022): Fundamental Climate Data Record of Microwave Imager Radiances, Edition 4, Satellite Application Facility on Climate Monitoring, DOI:10.5676/EUM_SAF_CM/FCDR_MWI/V004. Accessed: 2025-08-19.](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=FCDR_MWI_V004)   

*Land, ocean, coastline and ice* (LOCI) mask data [Stewart, J. S., Brodzik, M. J. & Scott, D. J. (2022). EASE Grids Ancillary Grid Information. (NSIDC-0772, Version 1). Projection: Northern hemisphere, Resolution: 3.125km, v1.1. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. DOI: 10.5067/GE8ET0MZ5ZVF.](https://nsidc.org/data/nsidc-0772/versions/1)   

Latitude and longitude coordinates data [Brodzik, M. J. & Knowles, K. (2011). EASE-Grid 2.0 Land-Ocean-Coastline-Ice Masks Derived from Boston University MODIS/Terra Land Cover Data. (NSIDC-0609, Version 1). Projection: Northern hemisphere, Resolution: 3.125km, Grid Dimensions (r x c): 5760x570. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. DOI: 10.5067/VY2JQZL9J8AQ.](https://nsidc.org/data/nsidc-0609/versions/1)   



## 2. Halo
The *halo method* takes the average of Envisat SIT around the pole hole and fills the pole hole with that average value.  
**Requires** LOCI mask and arrays of latitude and longitude coordinates in this folder.

### 2.1 Halo_vs_CS2.py
Calculates SIV inside the pole hole to validate against CS-2. 

### 2.2 Halo_EnvPeriod.py
Estimates SIV both inside and outside the pole hole. 



## 3. Ice_Age
The *ice age method* takes the average of Envisat SIT around the pole hole based on where *first-year ice* (FYI), *multi-year ice* (MYI) and ambiguous ice type exist. Then fills the pole hole with the certain average value where the corresponding ice type exist.  
**Requires** LOCI mask and arrays of latitude and longitude coordinates in this folder.

### 3.1 Ice_Age_vs_CS2.py
Calculates SIV inside the pole hole to validate against CS-2.

### 3.2 Ice_Age_EnvPeriod.py
Estimates SIV both inside and outside the pole hole.



## 4. Neural_Network
The neural network method uses passive microwave *brightness temperature* (TB) measurements, which allow coverage inside the pole hole, and a *neural network* (NN) model to predict SIT within the pole hole. The NN model has a  *Multilayer Perceptron* (MLP) architecture.  
**Requires** LOCI mask and arrays of latitude and longitude coordinates in this folder. 

### 4.1 format_data.py
Functions for formating and processing data products, used in majority of the other Python files in this folder.

### 4.2 TrainingData.py
Creates the training data, consists of TB and Envisat SIT products, used to train the NN model in NN.py.

### 4.3 NN.py
Trains the NN using PyTorch, used to create the NN model. 

#### 4.3.1 Automated_script.py
Optional: train NN automatically for 10 hidden nodes, each node is runned three times, with a certain amount of hidden layers (changed in the code by the user).  
**OBS!** if changing amount of hidden layers, it needs to be changed in NN.py

#### 4.3.2 hidden_layers_vs_nodes.py
Optional: plot RMSE, bias, coefcient of determination (R^2) for different amount of hidden layers and nodes.

#### 4.3.3 Inter_calibration_SSMI_SSMIS.py
Optional: check the difference in predicted SIT between SSM/I and SSMIS.

### 4.4 NN_vs_CS2.py
Calculates SIV inside the pole hole to validate against CS-2.  
**OBS!** Check that amount of hidden layers and nodes matches with the created NN model

### 4.5 NN_EnvPeriod.py
Estimates SIV both inside and outside the pole hole.   
**OBS!** Check that amount of hidden layers and nodes matches with the created NN model 



## 5. Estimating_SIV
Plotting the estimation of the total SIV in the whole Arctic region in the Envisat era.

### 5.1 Plot_estimatedSIV_all_methods.py
For all three methods: plots both total estimated SIV in Envisat era, from October 2002 to March 2012, and interanual change in each winter months, from October to April.

### 5.2 Envisat_vs_CS2.py
Plotting Envisat SIT products against CS-2 SIT products.
