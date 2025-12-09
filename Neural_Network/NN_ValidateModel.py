"""
File:       NN_ValidateModel_AlongTrack.py
Purpose:    Use the saved model from NN.py, input new TB data for a month to predict sea ice thickness (SIT) [m],
            calculate sea ice volume (SIV) [km^3] inside and outside the pole hole to get total Arctic SIV

Function:   synthetic_tracks, cell_area, format_SIT_outside, polehole

Other:      Created by Thea Jonsson 2025-09-26
"""

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
    def __init__(self, in_features=5, n_hidden=3, n_outputs=1):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        #self.hidden2 = nn.Linear(n_hidden, n_hidden)
        #self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()                       
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        #x = self.hidden2(x)
        #x = self.activation(x)
        #x = self.hidden3(x)
        #x = self.activation(x)
        x = self.output(x)          
        return x



"""
Function:   synthetic_tracks
Purpose:    Create synthetic satellite tracks in the pole hole (from lat_level to max_lat_level), 
            that are also masked a distance from land

Input:      NaN
Return:     x, y (float)
""" 
def synthetic_tracks(lat_level=81.5, max_lat_level=88, hemisphere="n"):
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

    water_mask = (land_data == 255) & (dist_km >= 50)

    lats = lats[::8, ::8]       
    lons = lons[::8, ::8]
    water_mask = water_mask[::8, ::8]

    lats_flat = lats.flatten()
    lons_flat = lons.flatten()
    water_flat = water_mask.flatten() 

    valid = (lats_flat >= lat_level) & (lats_flat <= max_lat_level) & water_flat
    lats_valid = lats_flat[valid]
    lons_valid = lons_flat[valid]

    x, y = lonlat_to_xy(lats_valid, lons_valid, hemisphere)

    return x, y



"""
Function:   format_SIC
Purpose:    Raed NetCDF file, loads data (lon, lat, SIC), convert lon/lat to x/y coordinates

Input:      file_paths (string)  
Return:     x_SIC, y_SIC (float)
            SIC (float): sea ice concentration (SIC) [%]
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
Purpose:    Uses format_SIC() to retrieve daily SIC data and coordinates, average the SIC values for a month 

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
        x_SIC, y_SIC, SIC = format_SIC(os.path.join(folder_path_SIC,files_SIC))
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
Function:   polehole
Purpose:    Predicts SIT inside the pole hole (using synthetic satellite tracks) with TB data as input to the NN model, 
            then calculates the SIV inside the pole hole and also the outisde using gridded EnviSat SIT data, 
            this is then added together for a total Arctic volume 
            Uses functions: synthetic_tracks, cell_area, format_SIT_outside
            Uses functions from format_data.py: land_mask, format_SSMIS, format_SIT, format_SIC
            Optional debug: plot 

Input:      year, month (string)
Return:     V_total (float)
"""
def polehole(year, month, debug=False):

    start_time = time.time()

    model = Model()
    #NN_model = torch.load(str(Path(__file__).resolve().parent.parent/"Data/NN/SSMIS_1month.pth")) 
    NN_model = torch.load("/Users/theajonsson/Desktop/2006_2007_80km/NN/h1n3/NN_Model.pth")
    model.load_state_dict(NN_model["model_state_dict"])
    scaler = NN_model["scaler"]

    x,y = synthetic_tracks()

    columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "X_SIT", "Y_SIT"]
    df_TB_SSMIS = pd.DataFrame(columns=columns) 
    index = 0
    df_TB_SSMIS["X_SIT"] = x
    df_TB_SSMIS["Y_SIT"] = y

    lons_valid, lats_valid, land_mask_data = fd.land_mask()

    # SSMIS
    if True:
        folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMIS/{year}/{month}/")
        files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

        y_eval_all = pd.DataFrame()
        day = 1
        for file_SSMIS in files_SSMIS:
            index = 0
            for i in range(len(group_SSMIS)):
                if i == 0:
                    vh = [1, 0, 2]  # Channel number: scene_env1 -> [V19, H19, V22]
                else:
                    vh = [1, 0]     # Channel number: scene_env2 -> [V37, H37]
                
                for j in range(len(vh)):
                    x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSMIS(x, y, os.path.join(folder_path_SSMIS,file_SSMIS), group_SSMIS[i], vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

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

    # Plot predicted SIT on polar map
    if debug:
        cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval_mean], cbar_label="Sea ice thickness [m]", save_name="PredictedSIT_inside")

    df = pd.DataFrame({
        "PredSIT": y_eval_mean,
        "X": TB_xy[:,0],
        "Y": TB_xy[:,1]
    })
    #df.to_csv("/Users/theajonsson/Desktop/Validate_PredSIT.csv", index=False)

    X = df["X"].values
    Y = df["Y"].values
    SIT_pred = df["PredSIT"]

    """
    file_CS2 = os.path.join(folder_CS2, year, f"ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{year}{month}-fv2.0.nc")
    x_CS2, y_CS2, SIT_CS2 = fd.format_SIT(file_CS2)
    tree = KDTree(list(zip(x_CS2.flatten(),y_CS2.flatten())))   # Gridded CS2

    X = df["X"].values
    Y = df["Y"].values
    SIT_pred = df["PredSIT"]
    distances, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Swath pred SIT (syntethic tracks)
    SIT_matched = SIT_CS2[indices]

    print(f"Mean predicted SIT: {np.nanmean(SIT_pred)} m")
    print(f"Mean predicted CS-2: {np.nanmean(SIT_matched)} m")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    # Plot histogram of CS-2 SIT and predicted SIT
    if debug:
        plt.figure()
        plt.hist(SIT_matched, bins=100, color="blue", alpha=0.5 ,edgecolor="black", zorder=3, label="CS-2 SIT", range=(SIT_matched.min(), SIT_matched.max()))
        plt.hist(SIT_pred, bins=100, color="red", edgecolor="black", zorder=1, label="Predicted SIT", range=(SIT_matched.min(), SIT_matched.max()))
        plt.xlabel("Sea Ice Thickness [m]")
        plt.ylabel("Amount [n]")
        plt.legend()
        #plt.savefig("/Users/theajonsson/Desktop/SIT_hist_2010nov.png", dpi=300, bbox_inches="tight") #
        plt.show()

    bias = np.mean(SIT_pred - SIT_matched)

    slope, intercept, r_value, p_value, std_err = linregress(SIT_matched, SIT_pred)
    r_squared = r_value**2

    rmse = mean_squared_error(SIT_matched, SIT_pred, squared=False)

    mean_env = np.mean(SIT_pred)
    mean_cs2 = np.mean(SIT_matched)

    # Scatter plot of CS-2 SIT (x-axis) vs pred SIT (y-axis)
    if debug:
        plt.figure()
        plt.scatter(SIT_matched, SIT_pred, s=10, color="#43a2ca", alpha=0.5, label=f"Bias: {bias:.3f} \nR-squared: {r_squared:.3f} \nRMSE={rmse:.3f}")
        plt.plot(SIT_matched, intercept + slope * SIT_matched, color="#8856a7", label="Fitted line")
        plt.plot([0,5],[0,5], color="black", linestyle="--", label="Optimal line")
        plt.scatter(mean_cs2, mean_env, color="#435fca", s=50, marker="x", zorder=10, label="Center of mass")
        plt.xlabel("CS-2 SIT [m]")
        plt.ylabel("Mean predicted SIT [m]")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 5)
        plt.xlim(0, 5)
        save_name = f"/Users/theajonsson/Desktop/{year}{month}.png"
        #plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.show()
    """


    # Volume inside of the pole hole
    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month)

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) # Gridded SIC
    _, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Syntethic tracks
    SIC_mean = SIC_mean[indices]

    area_cell = cell_area()

    # Calculate volume inside
    V_cell = ((SIT_pred*1e-3)*(SIC_mean/100))*area_cell
    V_cell[V_cell == 0] = np.nan
    V_tot = np.nansum(V_cell) 

    print(f"Volume inside the pole hole: {V_tot} [km^3]")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")



    # Plot predicted SIV on polar map
    if debug:
        name = f"{year}{month}_carto"
        cartoplot([X],[Y],[V_cell], cbar_label="Sea ice volume [km$^3$]", save_name=name)

    return V_tot



"""
Function:   format_SIT
Purpose:    Format file from the SIRAL instrument on the CryoSat-2 satellite
            Read NetCDF file, loads data (lon, lat, SIT), masks SIT to save positive values 
            from lat_level to max_lat_level, convert lon/lat to x/y coordinates

Input:      file_paths (string)
Return:     x_SIT, y_SIT, SIT (float)
"""
def format_CS2_SIT(file_path, lat_level=81.5, max_lat_level=88, hemisphere="n"):
    dataset = nc.Dataset(file_path, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()
    dataset.close()

    mask = (lat_SIT >= lat_level) & (lat_SIT <= max_lat_level) & (SIT >= 0)
    lat_SIT = lat_SIT[mask]
    lon_SIT = lon_SIT[mask]
    SIT = SIT[mask]

    x_SIT,y_SIT = lonlat_to_xy(lat_SIT, lon_SIT, hemisphere)

    X, Y = synthetic_tracks()

    tree = KDTree(list(zip(x_SIT.flatten(),y_SIT.flatten())))
    distances, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) 
    SIT = SIT[indices]    

    return X, Y, SIT



"""
Function:   volume_CS2
Purpose:    Calculates total Arctic volume [km^3] for each month using data from CryoSat-2
            Uses function: format_SIT, calc_SIC_mean, cell_area

Input:      year, month (string)
Return:     V_total 
"""
def volume_CS2(year, month, debug=False):

    file_sit = os.path.join(folder_CS2, year, f"ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{year}{month}-fv2.0.nc")
    x_SIT, y_SIT, SIT = format_CS2_SIT(file_sit)    # Unit: m
    print(f"Mean SIT: {np.nanmean(SIT)} m")

    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month)

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten())))
    distances, indices = tree.query(list(zip(x_SIT.flatten(),y_SIT.flatten()))) 
    SIC_mean = SIC_mean[indices]    # Unit: %

    area_cell = cell_area()     # Unit: km^2

    V_cell = ((SIT/1000)*(SIC_mean/100))*area_cell
    V_cell[V_cell == 0] = np.nan
    V_tot = np.nansum(V_cell)
    print(f"Total volume: {V_tot} km^3")



    # Plot SIV inside the pole hole using cartoplot
    if debug:
        cartoplot([x_SIT],[y_SIT],[V_cell], cbar_label="Sea ice volume [km$^3$]")


    return V_tot



data = {
    "2010": ["11", "12"],
    "2011": ["01", "02", "03", "04", "10", "11", "12"],
    "2012": ["01", "02", "03"]
}

folder_CS2 = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/")

# Estimated SIV inside the pole hole
if True:
    for year, months in data.items():
        for month in months:
            print(f"{year}-{month}")
            V_total = polehole(year, month)

            with open(str(Path(__file__).resolve().parent/"Total_volume_pred.txt"), "a") as file:
                file.write(f"{year}-{month}: {V_total}\n")
            
            #with open(str(Path(__file__).resolve().parent/"Results/Total_volume_EnvPeriod.txt"), "a") as file:
                #file.write(f"{year}-{month}: {V_total}\n")

# Calculate SIV for CS-2 inside the pole hole
if False:
    for year, months in data.items():
        for month in months:
            print(f"{year}-{month}")
            V_total = volume_CS2(year, month)

            with open(str(Path(__file__).resolve().parent/"Total_volume_cs2.txt"), "a") as file:
                file.write(f"{year}-{month}: {V_total}\n")



# Plot predicted SIV against CS2 SIV
if False:
    with open(str(Path(__file__).resolve().parent/"Total_volume_pred.txt"), "r") as f:
        pred_values = np.array([float(line.strip().split(":")[1]) for line in f])

    with open(str(Path(__file__).resolve().parent/"Total_volume_cs2.txt"), "r") as f:
        cs2_values = np.array([float(line.strip().split(":")[1]) for line in f])

    pred_1011 = pred_values[0:6]
    pred_1112 = pred_values[6:12]
    cs2_1011 = cs2_values[0:6]
    cs2_1112 = cs2_values[6:12]

    bias = np.mean(pred_values - cs2_values)
    slope, intercept, r_value, p_value, std_err = linregress(cs2_values, pred_values)
    print(slope)
    r_squared = r_value**2
    rmse = mean_squared_error(cs2_values, pred_values, squared=False)
    mean_pred = np.nanmean(pred_values)
    mean_cs2 = np.nanmean(cs2_values)

    plt.figure()
    plt.scatter(cs2_1011, pred_1011, color="#bae4bc", s=60, label="Nov 2010 - Apr 2011")
    plt.scatter(cs2_1112, pred_1112, color="#43a2ca", s=60, label="Oct 2011 - Mar 2012")
    plt.scatter([], [], color='none', label=f"RMSE={rmse:.3f}\nBias={bias:.3f}\nR$^2$={r_squared:.3f}")
    plt.scatter(mean_cs2, mean_pred, color="#810f7c", s=30, marker="*", zorder=10, label="Center of mass")
    plt.plot(cs2_values, intercept + slope * cs2_values, color="#810f7c", alpha=0.5, label="Fitted line")
    plt.plot([0, 10000], [0, 10000], color="black", linestyle="--", label="Optimal line")

    plt.xlabel("CS2 volume [km$^3$]")
    plt.ylabel("Predicted volume [km$^3$]")
    plt.xlim(0, 10000)
    plt.ylim(0, 10000)
    plt.grid(True)
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    dir = str(Path(__file__).resolve().parent/"Results/")
    plt.savefig(os.path.join(dir,"NN_Method.png"), dpi=300, bbox_inches="tight")
    plt.show()