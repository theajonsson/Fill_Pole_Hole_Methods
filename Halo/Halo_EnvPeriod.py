"""
File:       Averaging_all.py
Purpose:    Calculates total sea ice volume (SIV) [km^3] in Arctic, it first calculates inside and then outside 
            the pole hole, the volume inside the pole hole is based on average sea ice thickness (SIT) [m] 
            around the pole hole. This file is used when estimating the SIV for the whole Envisat period (2002-2012), 
            so the pole hole is defined from 81.5° N and upwards (90° N).

Function:   format_SIT_hole, format_SIT_outside, format_SIC, calc_SIC_mean, synthetic_tracks, cell_area, volume

Other:      Created by Thea Jonsson 2025-11-10
"""

import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.spatial import KDTree
from scipy.stats import linregress
from scipy.ndimage import distance_transform_edt
from ll_xy import lonlat_to_xy
from cartoplot import cartoplot



"""
Function:   land_mask
Purpose:    Create a mask of where water is with a distance from land and corresponding lon/lat coordiantes 

Input:      N/A
Return:     lons_valid, lats_valid (float): valid lon/lat coordinates where water is with a distance from land
            land_mask_data (uint8): mask array indicating only water pixels 
"""
def land_mask(min_distance_km=50, lat_level=60):

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
Function:   format_SIT_outside
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data (lat and SIT), masks SIT to save positive values 
            below max_lat_level and above lat_level

Input:      file_paths (string)
Return:     mask (boolean)
            SIT (float): sea ice thickness (SIT) [m]
"""
def format_SIT_outside(file_path, lons_valid, lats_valid, land_mask_data, lat_level=60, max_lat_level=81.5, hemisphere="n"):
    
    dataset = nc.Dataset(file_path, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()
    dataset.close()

    mask = (lat_SIT >= lat_level) & (lat_SIT < max_lat_level) & (SIT >= 0)
    lon_SIT = lon_SIT[mask]
    lat_SIT = lat_SIT[mask]
    SIT = SIT[mask]

    distances, nearest_coords, land_mask_data = nearest_neighbor(lon_SIT, lat_SIT, lons_valid, lats_valid, land_mask_data)

    x,y = lonlat_to_xy(nearest_coords[:,1], nearest_coords[:,0], hemisphere)

    land_mask = land_mask_data == 255
    x = x[land_mask]
    y = y[land_mask]
    SIT = SIT[land_mask]    

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
Purpose:    Calculates volume inisde and outside the pole hole, add them together for total Arctic volume for each month
            Uses function: format_SIT_hole, synthetic_tracks, calc_SIC_mean, cell_area, format_SIT_outside
            Optional debug: plot the SIV inside and outside the pole hole

Input:      year, month (string)
Return:     V_total (float)
"""
def volume(year, month, lons_valid, lats_valid, land_mask_data, debug=False):

    # Volume inside of the pole hole
    file_sit = os.path.join(folder_SIT, year, f"ESACCI-SEAICE-L3C-SITHICK-RA2_ENVISAT-NH25KMEASE2-{year}{month}-fv2.0.nc")
    average_SIT = format_SIT_hole(file_sit)
    print(f"Mean value of inside SIT: {average_SIT} m")

    X, Y = synthetic_tracks()

    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month) 

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) 
    _, indices = tree.query(list(zip(X.flatten(),Y.flatten()))) 
    SIC_in = SIC_mean[indices]    # Unit: %

    SIT = np.full(len(indices), average_SIT)    # Unit: m

    area_cell = cell_area()     # Unit: km^2

    V_cell_in = ((SIT/1000)*(SIC_in/100))*area_cell
    V_cell_in[V_cell_in == 0] = np.nan
    V_tot_in = np.nansum(V_cell_in)
    print(f"Volume inside the pole hole: {V_tot_in} km^3")



    # Volume outside of the pole hole
    x_SIT, y_SIT, SIT = format_SIT_outside(file_sit, lons_valid, lats_valid, land_mask_data)
    print(f"Mean value of outside SIT: {np.nanmean(SIT)} m")

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten()))) 
    _, indices = tree.query(list(zip(x_SIT.flatten(),y_SIT.flatten()))) 
    SIC_out = SIC_mean[indices]    # Unit: %

    V_cell_out = ((SIT/1000)*(SIC_out/100))*area_cell
    V_cell_out[V_cell_out == 0] = np.nan
    V_tot_out = np.nansum(V_cell_out)
    print(f"Volume outside the pole hole: {V_tot_out} km^3")



    # Total volume in Arctic
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
    "2002": ["10", "11", "12"],
    "2003": ["01", "02", "03", "04", "10", "11", "12"],
    "2004": ["01", "02", "03", "04", "10", "11", "12"],
    "2005": ["01", "02", "03", "04", "10", "11", "12"],
    "2006": ["01", "02", "03", "04", "10", "11", "12"],
    "2007": ["01", "02", "03", "04", "10", "11", "12"],
    "2008": ["01", "02", "03", "04", "10", "11", "12"],
    "2009": ["01", "02", "03", "04", "10", "11", "12"],
    "2010": ["01", "02", "03", "04", "10", "11", "12"],
    "2011": ["01", "02", "03", "04", "10", "11", "12"],
    "2012": ["01", "02", "03"]
}

folder_SIT = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")



lons_valid, lats_valid, land_mask_data = land_mask()        

for year, months in data.items():
    for month in months:
        print(f"{year}-{month}")
        V_total = volume(year, month, lons_valid, lats_valid, land_mask_data)

        with open(str(Path(__file__).resolve().parent.parent/"Estimating_SIV/HaloMethod_EnvPeriod.txt"), "a") as file:
            file.write(f"{year}-{month}: {V_total}\n")

        #with open(str(Path(__file__).resolve().parent/"Results/Total_volume_EnvPeriod.txt"), "a") as file:
        #    file.write(f"{year}-{month}: {V_total}\n")


# Should have a larger distance (don't include in Sweden)
# Set a fixed number on the bar in cartoplot (7) 
