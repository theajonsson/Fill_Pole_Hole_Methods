"""
File:       predSIV_vs_cs2SIV.py
Purpose:    Calculate total sea ice volume in Arctic using CryoSat-2 data
            Plots comparison of predicted SIV against CS-2 SIV

Function:   format_SIT, format_SIC, calc_SIC_mean, cell_area, volume

Other:      Created by Thea Jonsson 2025-10-16
"""
import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress



"""
Function:   format_SIT
Purpose:    Format file from the SIRAL instrument on the CryoSat-2 satellite
            Read NetCDF file, loads data (lon, lat, SIT), masks SIT to save positive values 
            from lat_level to max_lat_level, convert lon/lat to x/y coordinates

Input:      file_paths (string)
Return:     x_SIT, y_SIT, SIT (float)
"""
def format_SIT(file_path, lat_level=60, max_lat_level=88, hemisphere="n"):
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
    
    return x_SIT, y_SIT, SIT



"""
Function:   format_SIC
Purpose:    Read NetCDF file, loads data (lon, lat, SIC), convert lon/lat to x/y coordinates

Input:      file_paths (string)
Return:     x_SIC, y_SIC, SIC (float)
"""
def format_SIC(file_paths, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lat_SIC = np.array(dataset["lat"]).flatten()
    lon_SIC = np.array(dataset["lon"]).flatten()
    SIC = dataset["ice_conc"][:].filled(np.nan).flatten()    
    dataset.close()

    mask = ~np.isnan(SIC)
    SIC = SIC[mask]
    lat_SIC = lat_SIC[mask]
    lon_SIC = lon_SIC[mask]
    
    x_SIC,y_SIC = lonlat_to_xy(lat_SIC, lon_SIC, hemisphere)

    return x_SIC, y_SIC, SIC



"""
Function:   calc_SIC_mean
Purpose:    Uses format_SIC to retrieve daily SIC data and coordinates, average the SIC values for a month 

Input:      year, month (string)
Return:     x_SIC, y_SIC, SIC_mean (floats)
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
Purpose:    Calculates total Arctic volume [km^3] for each month using data from CryoSat-2
            Uses function: format_SIT, calc_SIC_mean, cell_area

Input:      year, month (string)
Return:     V_total 
"""
def volume(year, month):

    file_sit = os.path.join(folder_sit, year, f"ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{year}{month}-fv2.0.nc")
    x_SIT, y_SIT, SIT = format_SIT(file_sit)    # Unit: m
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

    return V_tot





data = {
    "2010": ["11", "12"],
    "2011": ["01", "02", "03", "04", "10", "11", "12"],
    "2012": ["01", "02", "03"]
}

folder_sit = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/")
folder_sic = str(Path(__file__).resolve().parent.parent/"Data/SIC/")

# Calculate volume for CS-2
if False:
    for year, months in data.items():
        for month in months:
            print(f"{year}-{month}")
            V_total = volume(year, month)

            with open(str(Path(__file__).resolve().parent/"Total_volume_cs2.txt"), "a") as file:
                file.write(f"{year}-{month}: {V_total}\n")





# Plot predicted SIV against CS2 SIV
if True:
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
    r_squared = r_value**2
    rmse = mean_squared_error(cs2_values, pred_values, squared=False)
    mean_pred = np.nanmean(pred_values)
    mean_cs2 = np.nanmean(cs2_values)

    plt.figure()
    plt.scatter(cs2_1011, pred_1011, color="#bae4bc", s=60, label="Nov 2010 - Apr 2011")
    plt.scatter(cs2_1112, pred_1112, color="#43a2ca", s=60, label="Oct 2010 - Mar 2012")
    plt.scatter(mean_cs2, mean_pred, color="#810f7c", s=30, marker="*", zorder=10, label="Center of mass")
    plt.plot(cs2_values, intercept + slope * cs2_values, color="#810f7c", alpha=0.5, label="Fitted line")
    plt.scatter([], [], color='none', label=f"RMSE={rmse:.3f}\nBias={bias:.3f}\nR$^2$={r_squared:.3f}")

    plt.plot([0, 30000], [0, 30000], color="black", linestyle="--", label="Optimal line")

    plt.xlabel("CS2 volume [km$^3$]")
    plt.ylabel("Predicted volume [km$^3$]")
    plt.xlim(0, 30000)
    plt.ylim(0, 30000)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("/Users/theajonsson/Desktop/MiddleMethod.png", dpi=300, bbox_inches="tight")
    plt.show()
    