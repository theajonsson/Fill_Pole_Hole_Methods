"""
File:       AveragingSIT_type.py
Purpose:    Calculate total sea ice volume (SIV) [km^3] inside the pole hole
            Where the volume inside the pole hole is based on average sea ice thickness (SIT) [m] for where different
            sea ice types exists. The ice types regards first-year ice (FYI), multi-year ice (MYI) and ambiguous ice type. 

Function:   format_type, calc_type_mean, nearest_neighbor, land_mask, format_SIT, format_SIC, calc_SIC_mean,
            cell_area, volume

Other:      Created by Thea Jonsson 2025-10-27
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
Function:   format_type
Purpose:    Read NetCDF file, loads data (lon, lat, ice type), convert lon/lat to x/y coordinates,
            mask ice type from lat_level up to max_lat_level and mask for desired ice type

Input:      file_paths (string)
Return:     x, y (float)
            fyi, myi, amb (float)
"""
def format_type(file_paths, lat_level=81.5, max_lat_level = 88, hemisphere="n"):
    
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon = np.array(dataset["lon"]).flatten()
    lat = np.array(dataset["lat"]).flatten() 
    ice_type = np.array(dataset["ice_type"]).flatten()
    dataset.close()

    x, y = lonlat_to_xy(lat, lon, hemisphere)

    first_year_ice = (lat >= lat_level) & (lat <= max_lat_level) & (ice_type == 2)    
    multi_year_ice = (lat >= lat_level) & (lat <= max_lat_level) & (ice_type == 3)
    ambiguous = (lat >= lat_level) & (ice_type == 4)
    fyi = np.where(first_year_ice, ice_type, np.nan)
    myi = np.where(multi_year_ice, ice_type, np.nan)
    amb = np.where(ambiguous, ice_type, np.nan)

    return x, y, fyi, myi, amb



"""
Function:   calc_type_mean
Purpose:    Uses format_type to retrieve daily ice type data and coordinates, 
            average the ice type data for a month, mask for NaN values 

Input:      year, month (string)
Return:     x_fyi, y_fyi (float)
            x_myi, y_myi (float)
            x_amb, y_amb (float)
"""
def calc_type_mean(year=2010, month=11):
 
    folder_path_type = str(Path(__file__).resolve().parent.parent/f"Data/SIType/{year}/{month}")
    files_type = sorted([
        f for f in os.listdir(folder_path_type)
        if f.startswith("ice_type_nh") and f[0].isalnum()
    ])
    fyi_all = pd.DataFrame()
    myi_all = pd.DataFrame()
    amb_all = pd.DataFrame()
    day = 1
    for files_type in files_type:
        x, y, fyi, myi, amb = format_type(os.path.join(folder_path_type,files_type))
        fyi_all[f"Day_{day}"] = fyi.flatten()
        myi_all[f"Day_{day}"] = myi.flatten()
        amb_all[f"Day_{day}"] = amb.flatten()
        day += 1

    fyi_mean = np.array(fyi_all.mean(axis=1))   
    myi_mean = np.array(myi_all.mean(axis=1))
    amb_mean = np.array(amb_all.mean(axis=1))

    mask_fyi = ~np.isnan(fyi_mean)
    mask_myi = ~np.isnan(myi_mean)
    mask_amb = ~np.isnan(amb_mean)

    x_fyi = x[mask_fyi]
    y_fyi = y[mask_fyi]
    x_myi = x[mask_myi]
    y_myi = y[mask_myi]
    x_amb = x[mask_amb]
    y_amb = y[mask_amb]

    return x_fyi, y_fyi, x_myi, y_myi, x_amb, y_amb



"""
Function:   nearest_neighbor
Purpose:    Find closest SIT value to one of the defined ice types

Input:      x (float): x coordinates for ice type and SIT
            y (float): y coordinates for ice type and SIT
            SIT (float)
Return:     distances (float): distance from each SIT point to its nearest ice type point
            nearest_SIT_coords (float): coordinates of that nearest SIT point
            SIT_data (float): corresponding SIT value at that nearest point
"""
def nearest_neighbor(x_type, y_type, x_SIT, y_SIT, SIT):

    type_coord = np.column_stack((x_type, y_type))
    SIT_coord = np.column_stack((x_SIT, y_SIT))

    tree = KDTree(SIT_coord)                         # K-Dimensional Tree on SIT coordinates 
    distances, indices = tree.query(type_coord)      # Queries the tree to find closest SIT coordinate point for each ice type coordinate point

    nearest_SIT_coords = SIT_coord[indices]           
    SIT_data = SIT[indices]                           

    return distances, nearest_SIT_coords, SIT_data



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

    # Distance from land
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
Function:   format_SIT
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data(lon, lat, SIT) and replaces fillvalue with NaN, 
            mask data to save everything from lat_level to max_lat_level and uses land_mask() and nearest_neighbor() 
            to mask a distance from land, converts lon/lat into x/y coordinates 

Input:      file_paths (string)
Return:     x_SIT, y_SIT (float)
            SIT (float): sea ice thickness (SIT) [m]
"""
def format_SIT(file_paths, lat_level=81.5, max_lat_level = 88, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()

    mask = (lat_SIT >= lat_level) & (lat_SIT <= max_lat_level) & (SIT >= 0)
    lat_SIT = lat_SIT[mask]
    lon_SIT = lon_SIT[mask]
    SIT = SIT[mask]

    lons_valid, lats_valid, land_mask_data = land_mask()
    _, nearest_coords, land_mask_data = nearest_neighbor(lon_SIT, lat_SIT, lons_valid, lats_valid, land_mask_data)

    x_SIT,y_SIT = lonlat_to_xy(nearest_coords[:,1], nearest_coords[:,0], hemisphere)

    mask = land_mask_data == 255
    x_SIT = x_SIT[mask]
    y_SIT = y_SIT[mask]
    SIT = SIT[mask]

    return x_SIT, y_SIT, SIT



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
Function:   volume
Purpose:    Calculates volume inisde the pole hole, add them together for total Arctic volume for each month
            Handles overlap between the different ice types, 
            Uses function: calc_type_mean, format_SIT, nearest_neighbor, calc_SIC_mean, cell_area
            Optional debug: plot the SIV inside the pole hole

Input:      year, month (string)
Return:     V_total (float)
"""
def volume(year, month, debug=False):

    # Average sea ice type: NaN pixels will be filled with a value if comming day has a value on that pixel
    x_fyi, y_fyi, x_myi, y_myi, x_amb, y_amb = calc_type_mean(year=year, month=month)

    file_sit = os.path.join(folder_sit, year, f"ESACCI-SEAICE-L3C-SITHICK-RA2_ENVISAT-NH25KMEASE2-{year}{month}-fv2.0.nc")
    x_SIT, y_SIT, SIT = format_SIT(file_sit)
    _, coords_fyi, SIT_fyi = nearest_neighbor(x_fyi, y_fyi, x_SIT, y_SIT, SIT)
    _, coords_myi, SIT_myi = nearest_neighbor(x_myi, y_myi, x_SIT, y_SIT, SIT)
    average_SIT_fyi = np.nanmean(SIT_fyi)
    average_SIT_myi = np.nanmean(SIT_myi)
    mean = ((average_SIT_fyi+average_SIT_myi)/2)

    print(f"Mean value of inside FYI SIT: {average_SIT_fyi} m")
    print(f"Mean value of inside MYI SIT: {average_SIT_myi} m")

    # Overlap between FYI, MYI and ambiguous ice type
    df_fyi = pd.DataFrame({"x": x_fyi,"y": y_fyi})
    df_myi = pd.DataFrame({"x": x_myi,"y": y_myi})
    df_amb = pd.DataFrame({"x": x_amb,"y": y_amb})

    # Overlap FYI o MYI
    overlap_1 = pd.merge(df_fyi, df_myi, on=["x", "y"], suffixes=("_x", "_y")) 
    overlapping_1 = overlap_1.copy()
    df1_filtered = df_fyi.merge(overlap_1[["x", "y"]], on=["x", "y"], how="left", indicator=True)
    df_fyi_filtered = df1_filtered[df1_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])
    df2_filtered = df_myi.merge(overlap_1[['x', 'y']], on=['x', 'y'], how='left', indicator=True)
    df_myi_filtered = df2_filtered[df2_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Overlap filtered FYI and ambiguous ice type
    overlap_2 = pd.merge(df_fyi_filtered, df_amb, on=["x", "y"], suffixes=("_x", "_y")) 
    overlapping_2 = overlap_2.copy()
    df1_filtered = df_fyi_filtered.merge(overlap_2[["x", "y"]], on=["x", "y"], how="left", indicator=True)
    df_fyi_filtered = df1_filtered[df1_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Overlap filtered MYI and ambiguous ice type
    overlap_3 = pd.merge(df_myi_filtered, df_amb, on=["x", "y"], suffixes=("_x", "_y")) 
    overlapping_3 = overlap_3.copy()
    df2_filtered = df_myi_filtered.merge(overlap_3[["x", "y"]], on=['x', 'y'], how='left', indicator=True)
    df_myi_filtered = df2_filtered[df2_filtered['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Contains any overlaps (F<->M, F<->amb, M<->amb)
    amb = pd.concat([overlapping_1, overlapping_2, overlapping_3], ignore_index=True) 

    df_fyi_filtered["SIT"] = average_SIT_fyi
    df_myi_filtered["SIT"] = average_SIT_myi
    amb["SIT"] = mean      

    total=pd.concat([df_fyi_filtered,df_myi_filtered,amb], ignore_index=True)
    X = np.array(total["x"])
    Y = np.array(total["y"])
    SIT = np.array(total["SIT"])



    # Volume for inside of the pole hole
    x_SIC, y_SIC, SIC_mean = calc_SIC_mean(year=year, month=month)

    tree = KDTree(list(zip(x_SIC.flatten(),y_SIC.flatten())))
    distances, indices = tree.query(list(zip(X.flatten(),Y.flatten())))
    SIC_mean = SIC_mean[indices]

    area_cell = cell_area()     

    V_cell = ((SIT/1000)*(SIC_mean/100))*area_cell
    V_cell[V_cell == 0] = np.nan
    V_tot = np.nansum(V_cell)
    print(f"Volume inside the pole hole: {V_tot} km^3")


    # Plot SIV both inside the pole hole using cartoplot
    if debug:
        name = f"{year}{month}_predicted"
        cartoplot([X],[Y],[V_cell], cbar_label="Sea ice volume [km$^3$]", save_name=name)


    return V_tot



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

folder_sit = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")
folder_SIT_CS2 = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/")

# Estimated SIV inside the pole hole
if False:
    for year, months in data.items():
        for month in months:
            print(f"{year}-{month}")
            V_total = volume(year, month)

            with open(str(Path(__file__).resolve().parent/"Total_volume_pred.txt"), "a") as file:
                file.write(f"{year}-{month}: {V_total}\n")

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
    plt.savefig(os.path.join(dir,"AveragingSIT_by_SeaIceType_Method.png"), dpi=300, bbox_inches="tight")
    plt.show()
