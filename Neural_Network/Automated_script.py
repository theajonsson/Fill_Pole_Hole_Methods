"""
File:       Automated_script.py
Purpose:    To automatically train NN by using NN_automated() in NN.py, 
            this is performed for 10 hidden nodes where each hidden node is run three times, 
            Optional: add one/more hidden layer(s), but then NN.py must also be altered! 
            the result of this is later used in hidden_layers_vs_nodes.py

Function:   synthetic_tracks, run, 

Other:      Created by Thea Jonsson 2025-09-26
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import format_data as fd 
from ll_xy import lonlat_to_xy
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.spatial import KDTree
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt
from NN import NN_automated



""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
""" ========================================================================================== """

# NN model using MLP architecture
# Input: brightness temperature (TB) [K] -> 5 different frequencies and polarization (V19, H19, V22, V37, H37)
# Output: sea ice thickness (SIT) [m]
class Model(nn.Module):
    def __init__(self, in_features=5, n_hidden=4, n_outputs=1):
        print(f"n_hidden = {n_hidden}")
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.hidden4 = nn.Linear(n_hidden, n_hidden)
        self.hidden5 = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()                       
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.hidden4(x)
        x = self.activation(x)
        x = self.hidden5(x)
        x = self.activation(x)
        x = self.output(x)          
        return x
"""
Adding one/more hidden layer(s) by:
self.hidden1 = nn.Linear(in_features, n_hidden)
self.hidden2 = nn.Linear(n_hidden, n_hidden)

x = self.hidden1(x)
x = self.activation(x)
x = self.hidden2(x)
x = self.activation(x)
"""



"""
Function:   synthetic_tracks
Purpose:    Create synthetic satellite tracks in the pole hole (from lat_level to max_lat_level), 
            that are also masked a distance from land

Input:      NaN
Return:     x, y (float)
""" 
def synthetic_tracks(lat_level=81.5, max_lat_level = 88, hemisphere="n"):
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
Function:   run
Purpose:    Predicts SIT inside the pole hole (using synthetic satellite tracks) with TB data as input to the NN model

Input:      n_hidden (int): number of hidden nodes
            x,y (float): x/y synthetic tracks coordinates
            tree (scipy.spatial.KDTree): used for nearest-neighbor queries
            SIT_CS2 (float): CryoSat-2 sea ice thickness (SIT) [m]
            lons_valid, lats_valid (float): valid lon/lat coordinates where water is with a distance from land
            land_mask_data (uint8): mask array indicating only water pixels
Return:     bias, r_squared, rmse (float)
""" 
def run(n_hidden, x, y, tree, SIT_CS2, lons_valid, lats_valid, land_mask_data):
    
    NN_model = NN_automated(n_hidden)
    model = Model(n_hidden=n_hidden)
    model.load_state_dict(NN_model["model_state_dict"])
    scaler = NN_model["scaler"]

    columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "X_SIT", "Y_SIT"]
    df_TB_SSMIS = pd.DataFrame(columns=columns) 
    index = 0
    df_TB_SSMIS["X_SIT"] = x
    df_TB_SSMIS["Y_SIT"] = y

    folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/"Data/TB_SSMIS/2010/11/")
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


    df = pd.DataFrame({
        "PredSIT": y_eval_mean,
        "X": TB_xy[:,0],
        "Y": TB_xy[:,1]
    })

    X = df["X"].values
    Y = df["Y"].values
    SIT_pred = df["PredSIT"]
    distances, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) # Swath pred SIT (syntethic tracks)
    SIT_matched = SIT_CS2[indices]

    
    # Important variables to evaluate the performance
    bias = np.mean(SIT_pred - SIT_matched)

    slope, intercept, r_value, p_value, std_err = linregress(SIT_matched, SIT_pred)
    r_squared = r_value**2

    rmse = mean_squared_error(SIT_matched, SIT_pred, squared=False)

    return bias, r_squared, rmse



if __name__ == "__main__":

    x,y = synthetic_tracks()

    lons_valid, lats_valid, land_mask_data = fd.land_mask()

    file_CS2 = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/2010/ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-201011-fv2.0.nc")
    x_CS2, y_CS2, SIT_CS2 = fd.format_SIT(file_CS2) # SIT, lon, lat .flatten()
    tree = KDTree(list(zip(x_CS2.flatten(),y_CS2.flatten())))   # Gridded CS2

    for n_hidden in range(10):

        for iteration in range(3):

            bias, r_squared, rmse = run(n_hidden+1, x, y, tree, SIT_CS2, lons_valid, lats_valid, land_mask_data)

            with open(str(Path(__file__).resolve().parent/"Evaluation_terms.txt"), "a") as file:
                file.write(f"({iteration+1}): Hidden nodes: {n_hidden+1}, bias: {bias}, R^2: {r_squared}, RMSE: {rmse} \n")
        
        with open(str(Path(__file__).resolve().parent/"Evaluation_terms.txt"), "a") as file:
            file.write("\n")
