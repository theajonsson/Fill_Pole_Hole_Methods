"""
File:       Averaging_all.py
Purpose:    Calculate total sea ice volume (SIV) [km^3] in Arctic by first calculate inside and then outside the pole hole
            Where the volume inside the pole hole is based on average sea ice thickness (SIT) [m] for where different
            sea ice types exists. The ice types regards first-year ice (FYI), multi-year ice (MYI) and ambiguous ice type. 

Function:   format_type, calc_type_mean, nearest_neighbor, land_mask, format_SIT, format_SIC, calc_SIC_mean,
            cell_area, format_SIT_outside, volume

Other:      Created by Thea Jonsson 2025-11-13
"""

import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
from ll_xy import lonlat_to_xy
from cartoplot import cartoplot
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt



"""
Function:   format_type
Purpose:    Read NetCDF file, loads data (lon, lat, ice type), convert lon/lat to x/y coordinates,
            mask ice type from lat_level up to max_lat_level and mask for desired ice type

Input:      file_paths (string)
Return:     x, y (float)
            fyi, myi, amb (float)
"""
def format_type(file_paths, lat_level=81.5, hemisphere="n"):
    
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon = np.array(dataset["lon"]).flatten()
    lat = np.array(dataset["lat"]).flatten() 
    ice_type = np.array(dataset["ice_type"]).flatten()
    dataset.close()

    x, y = lonlat_to_xy(lat, lon, hemisphere)

    first_year_ice = (lat >= lat_level) & (ice_type == 2)  
    multi_year_ice = (lat >= lat_level) & (ice_type == 3) 
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
def format_SIT(file_paths, lat_level=81.5, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()

    mask = (lat_SIT >= lat_level) & (SIT >= 0) 
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
Function:   volume
Purpose:    Calculates volume inisde and outside the pole hole, add them together for total Arctic volume for each month
            Handles overlap between the different ice types, 
            Uses function: calc_type_mean, format_SIT, nearest_neighbor, calc_SIC_mean, cell_area, format_SIT_outside
            Optional debug: plot the SIV inside and outside the pole hole

Input:      year, month (string)
Return:     V_total (float)
"""
def volume(year, month, lons_valid, lats_valid, land_mask_data, debug=False):

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
    SIC_in = SIC_mean[indices]

    area_cell = cell_area()     

    V_cell_in = ((SIT/1000)*(SIC_in/100))*area_cell
    V_cell_in[V_cell_in == 0] = np.nan
    V_tot_in = np.nansum(V_cell_in)
    print(f"Volume inside the pole hole: {V_tot_in} km^3")





    # Volume for outside of the pole hole
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

folder_sit = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")



lons_valid, lats_valid, land_mask_data = land_mask()

for year, months in data.items():
    for month in months:
        print(f"{year}-{month}")
        V_total = volume(year, month, lons_valid, lats_valid, land_mask_data)

        with open(str(Path(__file__).resolve().parent.parent/"Estimating_SIV/IceAgeMethod_EnvPeriod.txt"), "a") as file:
            file.write(f"{year}-{month}: {V_total}\n")

        #with open(str(Path(__file__).resolve().parent/"Results/Total_volume_EnvPeriod.txt"), "a") as file:
        #    file.write(f"{year}-{month}: {V_total}\n")
