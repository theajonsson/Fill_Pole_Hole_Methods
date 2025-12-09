# 2025-12-02

import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import format_data as fd 
from cartoplot import cartoplot, multi_cartoplot
from ll_xy import lonlat_to_xy
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.spatial import KDTree
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt



""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
""" ========================================================================================== """

# NN model using MLP architecture
# Input: brightness temperature (TB) [K] -> 5 different frequencies and polarization (V19, H19, V22, V37, H37)
# Output: sea ice thickness (SIT) [m]
class Model(nn.Module):
    def __init__(self, in_features=5, n_hidden=4, n_outputs=1):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()                       
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.output(x)          
        return x




"""
Function:   land_mask
Purpose:    Create a mask of where water is with a distance from land and corresponding lon/lat coordiantes 

Input:      N/A
Return:     lons_valid, lats_valid (float): valid lon/lat coordinates where water is with a distance from land
            land_mask_data (uint8): mask array indicating only water pixels 
"""
def land_mask(min_distance_km=50, lat_level=66):

    dataset = nc.Dataset(Path(__file__).resolve().parent/"NSIDC0772_LatLon_EASE2_N3.125km_v1.1.nc", "r", format="NETCDF4")
    lats = np.array(dataset["latitude"])
    lons = np.array(dataset["longitude"])
    dataset.close()

    grid_dir = Path(__file__).resolve().parent/"EASE2_N3.125km.LOCImask_land50_coast0km.5760x5760.bin"
    land_data = np.fromfile(grid_dir, dtype=np.uint8).reshape(5760, 5760)

    # Calculate distance from land
    land_binary = np.isin(land_data, [0, 101, 252]).astype(np.uint8)
    dist_pixels = distance_transform_edt(1 - land_binary)
    dist_km = dist_pixels * 3.125  # each pixel = 3.125 km

    # Mask water & distance from land: keep only water pixels (255) that are >= min_distance_km away from land
    water_mask = (land_data == 255) & (dist_km >= min_distance_km)

    lats_flat = lats.flatten()
    lons_flat = lons.flatten()
    water_flat = water_mask.flatten()

    valid = (~np.isnan(lats_flat)) & (lats_flat >= lat_level) & water_flat
    lats_valid = lats_flat[valid]
    lons_valid = lons_flat[valid]
    land_mask_data = np.full_like(lats_valid, 255, dtype=np.uint8) 

    return lons_valid, lats_valid, land_mask_data



"""
Function:   nearest_neighbor
Purpose:    Find closest coordinates to another set of coordinates, 
            set 1 is from where to match from and set 2 possible nearest neighbors to search among

Input:      lon (float): lon coordinates 
            lat (float): lat coordinates
            data (float)
Return:     distances (float): distance to each nearest neighbors
            nearest_coords (float): coordinates of these nearest neighbors
            data (float): data values on the nearest neighboring coordinates
"""
def nearest_neighbor(lon_1, lat_1, lon_2, lat_2, data):

    coord_1 = np.column_stack((lon_1, lat_1))
    coord_2 = np.column_stack((lon_2, lat_2))

    tree = KDTree(coord_2)                         
    distances, indices = tree.query(coord_1)      
    nearest_coords = coord_2[indices]         
    data = data[indices]                          

    return distances, nearest_coords, data



"""
Function:   format_SIT_outside
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data (lat and SIT), masks SIT to save positive values 
            below max_lat_level and above lat_level

Input:      file_paths (string)
Return:     mask (boolean)
            SIT (float): sea ice thickness (SIT) [m]
"""
def format_SIT_outside(file_path, lons_valid, lats_valid, land_mask_data, lat_level=66, max_lat_level=81.5, hemisphere="n"):
    
    dataset = nc.Dataset(file_path, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()
    dataset.close()

    mask = (lat_SIT >= lat_level) & (lat_SIT < max_lat_level) & (SIT >= 0)
    lon_SIT = lon_SIT[mask]
    lat_SIT = lat_SIT[mask]
    SIT = SIT[mask]

    x_SIT,y_SIT = lonlat_to_xy(lat_SIT, lon_SIT, hemisphere)
    x_valid,y_valid = lonlat_to_xy(lats_valid, lons_valid, hemisphere)

    distances, nearest_coords, land_mask_data = nearest_neighbor(x_SIT, y_SIT, x_valid, y_valid, land_mask_data)

    mask = distances < 10000        # If coord moved 10km it was probably on land and will be removed
    x = nearest_coords[:,0][mask]
    y = nearest_coords[:,1][mask]
    SIT = SIT[mask]

    return x, y, SIT



"""
Function:   synthetic_tracks
Purpose:    Create synthetic satellite tracks in the pole hole (from lat_level to max_lat_level), 
            that are also masked a distance from land

Input:      NaN
Return:     x, y (float)
""" 
def synthetic_tracks(lat_level=81.5, distance=50, hemisphere="n"):
    d = nc.Dataset(Path(__file__).resolve().parent/"NSIDC0772_LatLon_EASE2_N3.125km_v1.1.nc", "r", format="NETCDF4")
    lats = np.array(d["latitude"])
    lons = np.array(d["longitude"])
    d.close()

    grid_dir = Path(__file__).resolve().parent/"EASE2_N3.125km.LOCImask_land50_coast0km.5760x5760.bin"
    land_data = np.fromfile(grid_dir, dtype=np.uint8).reshape(5760, 5760)

    land_binary = np.isin(land_data, [0, 101, 252]).astype(np.uint8)

    # Distance from land
    dist_pixels = distance_transform_edt(1 - land_binary)
    dist_km = dist_pixels * 3.125  # each pixel = 3.125 km

    water_mask = (land_data == 255) & (dist_km >= distance)

    lats = lats[::8, ::8]       
    lons = lons[::8, ::8]
    water_mask = water_mask[::8, ::8]

    lats_flat = lats.flatten()
    lons_flat = lons.flatten()
    water_flat = water_mask.flatten() 

    valid = (lats_flat >= lat_level) & water_flat
    lats_valid = lats_flat[valid]
    lons_valid = lons_flat[valid]

    x, y = lonlat_to_xy(lats_valid, lons_valid, hemisphere)

    return x, y



"""
Function:   format_SIC
Purpose:    Raed NetCDF file, loads data (lon, lat, SIC), convert lon/lat to x/y coordinates

Input:      file_paths (string)  
Return:     x_SIC, y_SIC (float)
            SIC (float): sea ice concentration [%]
"""
def format_SIC(file_paths, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIC = np.array(dataset["lon"]).flatten()
    lat_SIC = np.array(dataset["lat"]).flatten()
    SIC = dataset["ice_conc"][:].filled(np.nan).flatten()     
    dataset.close()
    
    x_SIC,y_SIC = lonlat_to_xy(lat_SIC, lon_SIC, hemisphere)

    return x_SIC, y_SIC, SIC



"""
Function:   calc_SIC_mean
Purpose:    Uses format_SIC to retrieve daily SIC data and coordinates, average the SIC values for a month 

Input:      year, month (string)
Return:     x_SIC, y_SIC (float)
            SIC_mean (float)
"""
def calc_SIC_mean(year=2010, month=11):

    folder_path_SIC = str(Path(__file__).resolve().parent.parent/f"Data/SIC/{year}/{month}")
    files_SIC = sorted([
        f for f in os.listdir(folder_path_SIC)
        if f.startswith("ice_conc_nh") and f[0].isalnum()
    ])
    SIC_all = pd.DataFrame()
    day = 1
    for files_SIC in files_SIC:
        x_SIC, y_SIC, SIC = format_SIC(os.path.join(folder_path_SIC, files_SIC))
        SIC_all[f"Day_{day}"] = SIC.flatten()
        day += 1
    SIC_mean = np.array(SIC_all.mean(axis=1))

    mask = ~np.isnan(SIC_mean)
    SIC_mean = SIC_mean[mask]
    x_SIC = x_SIC[mask]
    y_SIC = y_SIC[mask]

    return x_SIC, y_SIC, SIC_mean



"""
Function:   cell_area
Purpose:    Calculates the area [km^2] for one grid cell 

Input:      NaN
Return:     area_cell (float)
"""
def cell_area():

    file_path = str(Path(__file__).resolve().parent.parent/"Data/SIC/2010/10/ice_conc_nh_ease2-250_cdr-v3p1_201010011200.nc")
    dataset = nc.Dataset(file_path, "r", format="NETCDF4")
    xc = dataset["xc"][:]
    yc = dataset["yc"][:]
    dx = np.diff(xc)
    dy = np.diff(yc)
    area_cell = abs(dx.mean()) * abs(dy.mean()) # Unit: km^2
    return area_cell



"""
Function:   volume
Purpose:    Predicts SIT inside the pole hole (using synthetic satellite tracks) with TB data as input to the NN model, 
            then calculates the SIV inside the pole hole and also the outisde using gridded EnviSat SIT data, 
            this is then added together for a total Arctic volume 
            Uses functions: synthetic_tracks, cell_area, format_SIT_outside
            Uses functions from format_data.py: land_mask, format_SSMIS, format_SIT, format_SIC
            Optional debug: plot 

Input:      year, month (string)
Return:     V_total (float)
"""
def volume(year, month, lons_valid, lats_valid, land_mask_data, debug=False):

    start_time = time.time()

    model = Model()
    #NN_model = torch.load(str(Path(__file__).resolve().parent.parent/"Data/NN/SSMIS_1month.pth")) 
    NN_model = torch.load("/Users/theajonsson/Desktop/2006_2007_80km/NN/h2n4/NN_Model.pth")
    model.load_state_dict(NN_model["model_state_dict"])
    scaler = NN_model["scaler"]

    x,y = synthetic_tracks()

    columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "X_SIT", "Y_SIT"]
    df_TB = pd.DataFrame(columns=columns) 
    index = 0
    df_TB["X_SIT"] = x
    df_TB["Y_SIT"] = y

    year_int = int(year)
    if 2002 <= year_int <= 2005 or (year == "2006" and month in ["01", "02", "03", "04"]):
        sensor = "SSMI"
    else:
        sensor = "SSMIS"
    print(f"Using sensor:{sensor} for {year}-{month}")

    folder = Path(__file__).resolve().parent.parent
    if sensor == "SSMI":
        folder = folder / f"Data/TB_SSMI/{year}/{month}/"
        tb_order = list(range(5))    
    else:
        folder = folder / f"Data/TB_SSMIS/{year}/{month}/"
        tb_order = [
            ("scene_env1", 1),  # 19V
            ("scene_env1", 0),  # 19H
            ("scene_env1", 2),  # 22V
            ("scene_env2", 1),  # 37V
            ("scene_env2", 0)   # 37H
        ]
    files = sorted([f for f in os.listdir(folder) if f[0].isalnum()])

    y_eval_all = pd.DataFrame()
    day = 1
    for file in files:
        index = 0

        if sensor == "SSMI":
            for vh in tb_order:
                _, _, _, TB_freq, _ = fd.format_SSM_I(x, y, os.path.join(folder, file), group_SSM_I, vh, lons_valid, lats_valid, land_mask_data)
                df_TB[columns[index]] = TB_freq
                index += 1

        elif sensor == "SSMIS":
            for group, channel in tb_order:
                _, _, _, TB_freq, _ = fd.format_SSMIS(x, y, os.path.join(folder, file), group, channel, lons_valid, lats_valid, land_mask_data)
                df_TB[columns[index]] = TB_freq
                index += 1
        
        Test_TB = df_TB[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
        TB_xy =  df_TB[["X_SIT","Y_SIT"]].values 

        Test_TB = scaler.fit_transform(Test_TB)
        Test_TB = torch.FloatTensor(Test_TB)

        with torch.no_grad():
            y_eval = model(Test_TB)

        y_eval_all[f"Day_{day}"] = y_eval.squeeze().numpy()
        print(f"Day {day} done")
        day += 1

    y_eval_mean = np.array(y_eval_all.mean(axis=1))

    # SSM/I
    if False:
        folder_path_SSMI = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMI/{year}/{month}/")
        files_SSMI = sorted([f for f in os.listdir(folder_path_SSMI) if f[0].isalnum()])

        y_eval_all = pd.DataFrame()
        day = 1
        for file_SSMI in files_SSMI:
            index = 0
            vh = [0, 1, 2, 3, 4]
        
            for j in range(len(vh)):
                x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSM_I(x, y, os.path.join(folder_path_SSMI,file_SSMI), group_SSM_I, vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

                df_TB_SSMIS[columns[index]] = TB_freq     
                index += 1

            Test_TB = df_TB_SSMIS[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
            TB_xy =  df_TB_SSMIS[["X_SIT","Y_SIT"]].values 

            Test_TB = scaler.fit_transform(Test_TB)
            Test_TB = torch.FloatTensor(Test_TB)

            with torch.no_grad():
                y_eval = model.forward(Test_TB)
            
            y_eval_all[f"Day_{day}"] = y_eval.squeeze().numpy()
            print(f"Day_{day} done")
            day += 1
        y_eval_mean = np.array(y_eval_all.mean(axis=1))

    # SSMIS
    if False:
        folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMIS/{year}/{month}/")
        files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

        y_eval_all = pd.DataFrame()
        day = 1
        for file_SSMIS in files_SSMIS:
            index = 0
            tb_order = [
                ("scene_env1", 1),   # 19V
                ("scene_env1", 0),   # 19H
                ("scene_env1", 2),   # 22V
                ("scene_env2", 1),   # 37V
                ("scene_env2", 0)    # 37H
            ]
            for group, channel in tb_order:                
                x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSMIS(x, y, os.path.join(folder_path_SSMIS,file_SSMIS), group, channel, lons_valid, lats_valid, land_mask_data, debug=False)
                df_TB_SSMIS[columns[index]] = TB_freq     
                index += 1

            Test_TB = df_TB_SSMIS[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
            TB_xy =  df_TB_SSMIS[["X_SIT","Y_SIT"]].values 

            Test_TB = scaler.fit_transform(Test_TB)
            Test_TB = torch.FloatTensor(Test_TB)

            with torch.no_grad():
                y_eval = model.forward(Test_TB)
            
            y_eval_all[f"Day_{day}"] = y_eval.squeeze().numpy()
            print(f"Day_{day} done")
            day += 1
        y_eval_mean = np.array(y_eval_all.mean(axis=1))

    df = pd.DataFrame({
        "PredSIT": y_eval_mean,
        "X": TB_xy[:,0],
        "Y": TB_xy[:,1]
    })

    X = df["X"].values
    Y = df["Y"].values
    SIT_pred = df["PredSIT"]
   


    # Volume inside of the pole hole
    print(f"Mean value of inside predicted SIT: {np.nanmean(SIT_pred)} m")

    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month)

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) # Gridded SIC
    _, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Syntethic tracks
    SIC_in = SIC_mean[indices]

    area_cell = cell_area()

    # Calculate volume inside
    V_cell_in = ((SIT_pred*1e-3)*(SIC_in/100))*area_cell
    V_cell_in[V_cell_in == 0] = np.nan
    V_tot_in = np.nansum(V_cell_in) 
    print(f"Volume inside the pole hole: {V_tot_in} [km^3]")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    

    # Volume outside of the pole hole
    file_sit = os.path.join(folder_SIT, year, f"ESACCI-SEAICE-L3C-SITHICK-RA2_ENVISAT-NH25KMEASE2-{year}{month}-fv2.0.nc")
    x_SIT, y_SIT, SIT = format_SIT_outside(file_sit, lons_valid, lats_valid, land_mask_data)
    print(f"Mean value of outside SIT: {np.nanmean(SIT)} m")

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) 
    _, indices = tree.query(list(zip(x_SIT.flatten(),y_SIT.flatten()))) 
    SIC_out = SIC_mean[indices]    

    V_cell_out = ((SIT/1000)*(SIC_out/100))*area_cell
    V_cell_out[V_cell_out == 0] = np.nan
    V_tot_out = np.nansum(V_cell_out)
    print(f"Volume outside the pole hole: {V_tot_out} km^3")



    # Total volume in the Arctic Ocean
    V_total = V_tot_in + V_tot_out
    print(f"Total volume in Arctic is: {V_total} km^3 \n")



    # Plot SIV both inside and outside pole hole using cartoplot
    if debug:
        x_SIC = x_SIC[indices]
        y_SIC = y_SIC[indices]

        name = f"{year}{month}_carto"
        cartoplot([X, x_SIC],[Y, y_SIC],[V_cell_in, V_cell_out], cbar_label="Sea ice volume [km$^3$]", save_name=name)

    return V_total



data = {
    #"2002": ["10", "11", "12"],
    #"2003": ["01", "02", "03", "04", "10", "11", "12"],
    #"2004": ["01", "02", "03", "04", "10", "11", "12"],
    #"2005": ["01", "02", "03", "04", "10", "11", "12"],
    "2006": ["01", "02", "03", "04", "10", "11", "12"],
    #"2007": ["01", "02", "03", "04", "10", "11", "12"],
    #"2008": ["01", "02", "03", "04", "10", "11", "12"],
    #"2009": ["01", "02", "03", "04", "10", "11", "12"],
    #"2010": ["01", "02", "03", "04", "10", "11", "12"],
    #"2011": ["01", "02", "03", "04", "10", "11", "12"],
    #"2012": ["01", "02", "03"]
}

folder_SIT = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")



lons_valid, lats_valid, land_mask_data = land_mask()        

for year, months in data.items():
    for month in months:
        print(f"{year}-{month}")
        V_total = volume(year, month, lons_valid, lats_valid, land_mask_data)

        #with open(str(Path(__file__).resolve().parent/"Total_volume_pred_EnvPeriod.txt"), "a") as file:
        #        file.write(f"{year}-{month}: {V_total}\n")

        #with open(str(Path(__file__).resolve().parent.parent/"Estimating_SIV/NNMethod_EnvPeriod.txt"), "a") as file:
        #    file.write(f"{year}-{month}: {V_total}\n")

        #with open(str(Path(__file__).resolve().parent/"Results/Predicted_SIV_EnvPeriod/Total_volume_EnvPeriod.txt"), "a") as file:
        #    file.write(f"{year}-{month}: {V_total}\n")