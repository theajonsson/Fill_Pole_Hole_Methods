"""
File:       AveragingSIT.py
Purpose:    Calculate sea ice volume (SIV) [km^3] inside the pole hole.
            The volume inside the pole hole is based on average sea ice thickness (SIT) [m] around the pole hole.
            This file is used when comparing the three different methods, so the pole hole is defined from 81.5° N to 88° N.

Function:   format_SIT_hole, synthetic_tracks, format_SIC, calc_SIC_mean, cell_area, volume, format_CS2_SIT, volume_CS2

Other:      Created by Thea Jonsson 2025-10-16
"""

import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.ndimage import distance_transform_edt
from ll_xy import lonlat_to_xy
from cartoplot import cartoplot



"""
Function:   format_SIT_hole
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data (lat, SIT), masks SIT to save positive values from lat_level to max_lat  
            and average all these values 

Input:      file_paths (string)
Return:     average_SIT (float)
"""
def format_SIT_hole(file_paths, lat_level=75, max_lat=81.5, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()

    mask = (lat_SIT >= lat_level) & (lat_SIT <= max_lat) & (SIT >= 0)
    SIT = SIT[mask]

    average_SIT = np.nanmean(SIT)

    return average_SIT



"""
Function:   synthetic_tracks
Purpose:    Create synthetic satellite tracks in the pole hole (from lat_level to max_lat_level), 
            that are also masked a distance from land

Input:      NaN
Return:     x, y (float)
""" 
def synthetic_tracks(lat_level=81.5, max_lat_level = 88, distance=50, hemisphere="n"):
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
Purpose:    Calculates SIV, with an average SIT value, inisde the pole hole for each month using synthetic tracks
            Uses function: format_SIT_hole, synthetic_tracks, calc_SIC_mean, cell_area
            Optional debug: plot the SIV inside the pole hole

Input:      year, month (string)
Return:     V_total (float)
"""
def volume(year, month, debug=False):

    file_sit = os.path.join(folder_SIT, year, f"ESACCI-SEAICE-L3C-SITHICK-RA2_ENVISAT-NH25KMEASE2-{year}{month}-fv2.0.nc")
    average_SIT = format_SIT_hole(file_sit)
    print(f"Mean value of inside SIT: {average_SIT} m")

    X, Y = synthetic_tracks()

    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month) 

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) 
    _, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) 
    SIC_mean = SIC_mean[indices]    # Unit: %

    SIT = np.full(len(indices), average_SIT)    # Unit: m

    area_cell = cell_area()     # Unit: km^2

    V_cell_in = ((SIT/1000)*(SIC_mean/100))*area_cell
    V_cell_in[V_cell_in == 0] = np.nan
    V_tot = np.nansum(V_cell_in)
    print(f"Volume inside the pole hole: {V_tot} km^3")


    # Plot SIV inside the pole hole using cartoplot
    if debug:
        name = f"{year}{month}_predicted"
        cartoplot([X],[Y],[V_cell_in], cbar_label="Sea ice volume [km$^3$]", save_name=name)

    return V_tot



"""
Function:   format_SIT
Purpose:    Format file from the SIRAL instrument on the CryoSat-2 satellite
            Read NetCDF file, loads data (lon, lat, SIT), masks SIT to save positive values 
            from lat_level to max_lat_level, convert lon/lat to x/y coordinates,
            finds nearest-neighbouring SIT values to where synthetic track is

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
Purpose:    Calculates SIV [km^3] inside the pole hole for each month using data from CryoSat-2
            Uses function: format_SIT, calc_SIC_mean, cell_area

Input:      year, month (string)
Return:     V_total 
"""
def volume_CS2(year, month, debug=False):

    file_sit = os.path.join(folder_SIT_CS2, year, f"ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{year}{month}-fv2.0.nc")
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
        name = f"{year}{month}_CS2"
        cartoplot([x_SIT],[y_SIT],[V_cell], cbar_label="Sea ice volume [km$^3$]", save_name=name)


    return V_tot



data = {
    "2010": ["11", "12"],
    "2011": ["01", "02", "03", "04", "10", "11", "12"],
    "2012": ["01", "02", "03"]
}
folder_SIT = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")
folder_SIT_CS2 = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/")

# Calculates predicetd SIV inside the pole hole
if False:
    for year, months in data.items():
        for month in months:
            print(f"{year}-{month}")
            V_total = volume(year, month)

            with open(str(Path(__file__).resolve().parent/"Total_volume_pred.txt"), "a") as file:
                file.write(f"{year}-{month}: {V_total}\n")

# Calculates SIV for CS-2 inside the pole hole
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
    r_squared = r_value**2
    rmse = mean_squared_error(cs2_values, pred_values, squared=False)
    mean_pred = np.nanmean(pred_values)
    mean_cs2 = np.nanmean(cs2_values)

    plt.figure()
    plt.scatter(cs2_1011, pred_1011, color="#bae4bc", s=60, label="Nov 2010 - Apr 2011")
    plt.scatter(cs2_1112, pred_1112, color="#43a2ca", s=60, label="Oct 2011 - Mar 2012")
    plt.scatter(mean_cs2, mean_pred, color="#810f7c", s=30, marker="*", zorder=10, label="Center of mass")
    plt.plot(cs2_values, intercept + slope * cs2_values, color="#810f7c", alpha=0.5, label="Fitted line")
    plt.scatter([], [], color='none', label=f"RMSE={rmse:.3f}\nBias={bias:.3f}\nR$^2$={r_squared:.3f}")

    plt.plot([0, 10000], [0, 10000], color="black", linestyle="--", label="Optimal line")

    plt.xlabel("CS2 volume [km$^3$]")
    plt.ylabel("Predicted volume [km$^3$]")
    plt.xlim(0, 10000)
    plt.ylim(0, 10000)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    dir = str(Path(__file__).resolve().parent/"Results/")
    plt.savefig(os.path.join(dir,"AveragingSIT_Method.png"), dpi=300, bbox_inches="tight")
    plt.show()
