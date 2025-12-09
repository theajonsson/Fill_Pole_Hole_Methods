"""
File:       format_data.py
Purpose:    Provides functions for formating data and performance of necessary steps for further analysis
            Uses file ll_xy.py

Function:   nearest_neighbor, land_mask, format_SIT, format_SSM_I, format_SSMIS, format_SIC

Other:      Created by Thea Jonsson 2025-08-28
"""

from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from cartoplot import cartoplot



"""
Function:   nearest_neighbor
Purpose:    Find closest TB data point to each SIT data point

Input:      x (float): x coordinates for SIT and TB
            y (float): y coordinates for SIT and TB
            TB (float)
Return:     distances (float): distance from each SIT point to its nearest TB point
            nearest_TB_coords (float): coordinates of that nearest TB point
            TB_freq (float): corresponding TB value (for choosen frequency) at that nearest point
"""
def nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB):

    TB_coord = np.column_stack((x_TB, y_TB))
    SIT_coord = np.column_stack((x_SIT, y_SIT))

    tree = KDTree(TB_coord)                         # K-Dimensional Tree on TB coordinates 
    distances, indices = tree.query(SIT_coord)      # Queries the tree to find closest TB coordinate point for each SIT coordinate point

    nearest_TB_coords = TB_coord[indices]         
    TB_data = TB[indices]                          

    return distances, nearest_TB_coords, TB_data



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
    land_mask_data = water_flat[valid]
    #land_mask_data = np.full_like(lats_valid, 255, dtype=np.uint8) 

    return lons_valid, lats_valid, land_mask_data



"""
Function:   format_SIT
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data(lon, lat, SIT) and replaces fillvalue with NaN, 
            mask data to save everything from lat_level and above, uses land_mask() and nearest_neighbor() 
            to mask a distance from land, converts lon/lat into x/y coordinates 

Input:      file_paths (string)
Return:     x_SIT, y_SIT (float)
            SIT (float): sea ice thickness (SIT), unit: [m]
"""
def format_SIT(file_paths, lat_level=81.5, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()

    """
    x_SIT,y_SIT = lonlat_to_xy(lat_SIT, lon_SIT, hemisphere)
    dx = np.diff(x_SIT)
    dy = np.diff(y_SIT)
    distances = np.sqrt(dx**2 + dy**2)
    dist_cum = np.insert(np.cumsum(distances), 0, 0)

    mask = (dist_cum >= 0) & (dist_cum <= 50000)  
    dist_sub = dist_cum[mask]
    dist_plot = dist_sub - dist_sub[0]
    SIT_sub = SIT[mask]

    plt.figure(figsize=(10,5))
    plt.plot(dist_plot/1000, SIT_sub, "-b")
    plt.xlabel("Distance along track [km]")
    plt.ylabel("Sea ice thickness [m]")
    plt.ylim(0,10)
    plt.xlim(0,50)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/Users/theajonsson/Desktop/EnvisatSIT_200703_1750_1800.png", dpi=300, bbox_inches="tight")
    plt.show()
    breakpoint()
    """

    #mask = (lat_SIT >= lat_level) & (SIT >= 0)
    mask = (lat_SIT >= lat_level) & ((SIT >= 0) | np.isnan(SIT))
    lat_SIT = lat_SIT[mask]
    lon_SIT = lon_SIT[mask]
    SIT = SIT[mask]

    lons_valid, lats_valid, land_mask_data = land_mask()

    x_SIT,y_SIT = lonlat_to_xy(lat_SIT, lon_SIT, hemisphere)
    x_valid,y_valid = lonlat_to_xy(lats_valid, lons_valid, hemisphere)
 
    distances, nearest_coords, land_mask_data = nearest_neighbor(x_SIT, y_SIT, x_valid, y_valid, land_mask_data)

    mask = distances < 10000        
    x_SIT = nearest_coords[:,0][mask]
    y_SIT = nearest_coords[:,1][mask]
    SIT = SIT[mask]

    return x_SIT, y_SIT, SIT







"""
Function:   format_SSM_I
Purpose:    Format file from the SSM/I instrument on the DMSP-F14 satellite
            Read NetCDF file, loads data (lon, lat, TB) and replaces fillvalue with NaN, mask data 
            to save everything from lat_level and above and removes NaN, uses land_mask() and nearest_neighbor() 
            to mask a distance from land, converts lon/lat into x/y coordinates, uses function nearest_neighbor() 
            to find nearest TB points to each SIT points
            Optional debug: plot TB, SIT and nearest TB to see if the coordinates match up

Input:      x_SIT, y_SIT (float)
            file_paths (string)
            group (string)
            channel (int)
            lons_valid, lats_valid (float)
            land_mask_data (unit8)
Return:     x_TB (float)
            y_TB (float)
            TB (float): latitude and land masked TB values
            TB_near (float): nearest TB values in realationship to the SIT values
            nearest_TB_coords (float, (N,2)-array)
"""
def format_SSM_I(x_SIT, y_SIT, file_paths, group, channel, lons_valid, lats_valid, land_mask_data,
                 lat_level=66, hemisphere="n", debug=False):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())      # NaN instead of _FillValue=-9e+33          
    ical = np.array(dataset.groups[group].variables["ical"][:,channel,:].filled(np.nan).flatten())
    dataset.close()

    TB = TB + ical

    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask] 

    x_TB,y_TB = lonlat_to_xy(lat_TB, lon_TB, hemisphere)

    distances, nearest_TB_coords, TB_near = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)

    # Plot positions of nearest_TB_coords to check if they line up closely to the positions of SITs
    if debug:
        print("Group: ", group)
        print("Channel: ", channel)
        print("Mean distance:", np.mean(distances))
        print("Max distance:", np.max(distances))
        print("Min distance:", np.min(distances))

        plt.scatter(x_TB, y_TB, s=0.5, c="blue", alpha=0.2, label="TB")
        plt.scatter(x_SIT, y_SIT, s=15, c="red", label="SIT")
        plt.scatter(nearest_TB_coords[:, 0], nearest_TB_coords[:, 1], s=5, c="green", label="Nearst TB")
        plt.legend(loc="upper right")
        plt.title("SIT and their nearest TB neighbor")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        #plt.savefig("/Users/theajonsson/Desktop/nearestTB.png", dpi=300, bbox_inches="tight")
        plt.show()

    return x_TB, y_TB, TB, TB_near, nearest_TB_coords





"""
Function:   format_SSMIS
Purpose:    Format file from the SSMIS instrument on the DMSP-F16 satellite
            Read NetCDF file, loads data (lon, lat, TB) and replaces fillvalue with NaN, mask data 
            to save everything from lat_level and above and removes NaN, uses land_mask() and nearest_neighbor() 
            to mask a distance from land, converts lon/lat into x/y coordinates, uses function nearest_neighbor() 
            to find nearest TB points to each SIT points
            Optional debug: plot TB, SIT and nearest TB to see if the coordinates match up   

Input:      x_SIT, y_SIT (float)
            file_paths (string)
            group (string)
            channel (int)
            lons_valid, lats_valid (float)
            land_mask_data (unit8)
Return:     x_TB (float)
            y_TB (float)
            TB (float): latitude and land masked TB values
            TB_near (float): nearest TB values in realationship to the SIT values
            nearest_TB_coords (float, (N,2)-array)
"""
def format_SSMIS(x_SIT, y_SIT, file_paths, group, channel, lons_valid, lats_valid, land_mask_data,
                 lat_level=66, hemisphere="n", debug=False):
    
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()   
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())      # NaN instead of _FillValue=-9e+33
    ical = np.array(dataset.groups[group].variables["ical"][:,channel,:].filled(np.nan).flatten())
    dataset.close()

    TB = TB + ical

    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask]

    x_TB,y_TB = lonlat_to_xy(lat_TB, lon_TB, hemisphere)
    #x_valid,y_valid = lonlat_to_xy(lats_valid, lons_valid, hemisphere)

    #distances, nearest_coords, land_mask_data = nearest_neighbor(x_TB, y_TB, x_valid, y_valid, land_mask_data)

    #mask = distances < 10000        
    #x_TB = nearest_coords[:,0][mask]
    #y_TB = nearest_coords[:,1][mask]
    #TB = TB[mask]

    distances, nearest_TB_coords, TB_near = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)



    # Plot positions of nearest_TB_coords to check if they line up closely to the positions of SITs
    if debug:
        print("Group: ", group)
        print("Channel: ", channel)
        print("Mean distance:", np.mean(distances))
        print("Max distance:", np.max(distances))
        print("Min distance:", np.min(distances))

        plt.scatter(x_TB, y_TB, s=0.5, c="blue", alpha=0.2, label="TB")
        plt.scatter(x_SIT, y_SIT, s=15, c="red", label="SIT")
        plt.scatter(nearest_TB_coords[:, 0], nearest_TB_coords[:, 1], s=5, c="green", label="Nearst TB")
        plt.legend(loc="upper right")
        plt.title("SIT and their nearest TB neighbor")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        name = f"/Users/theajonsson/Desktop/nearestTB_{group}_{channel}.png"
        plt.savefig(name, dpi=300, bbox_inches="tight")

    return x_TB, y_TB, TB, TB_near, nearest_TB_coords




"""
Function:   format_SIC
Purpose:    Read NetCDF file, loads data (lon, lat, SIC) and replaces fillvalue with NaN, 
            filter all data to save data from lat_level to max_lat_level and removes NaN, 
            converts lon/lat into x/y coordinates 

Input:      file_paths (string)
Return:     x_SIC, y_SIC (float)
            SIC (float): sea ice concentration (SIC), unit: [%]
"""
"""
def format_SIC(file_paths, lat_level=81.5, max_lat_level = 88, hemisphere="n"):
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lat_SIC = np.array(dataset["lat"]).flatten()
    lon_SIC = np.array(dataset["lon"]).flatten()
    SIC = dataset["ice_conc"][:].filled(np.nan).flatten() 
    dataset.close()

    mask = (lat_SIC >= lat_level) & (lat_SIC <= max_lat_level)
    lat_SIC = lat_SIC[mask]
    lon_SIC = lon_SIC[mask]
    SIC = SIC[mask]

    x_SIC,y_SIC = lonlat_to_xy(lat_SIC, lon_SIC, hemisphere)

    return x_SIC, y_SIC, SIC
"""

"""
Function:   split_tracks
Purpose:    

Input:      
Return:     
"""
def split_tracks(df, distance_segment = 50000):

  # Calculate distance between data points
  x = df["X_SIT"].values
  y = df["Y_SIT"].values
  dx = np.diff(x)
  dy = np.diff(y)
  distances = np.sqrt(dx**2 + dy**2)
  distances = np.insert(distances, 0, 0)  # First data point: distance = 0

  df_seg = pd.DataFrame({
          "SIT": [],
          "X_SIT": [],
          "Y_SIT": [],
          "TB_V19": [],
          "TB_H19": [],
          "TB_V22": [],
          "TB_V37": [],
          "TB_H37": []
        })

  temp = {
      "TB_V19": [], "TB_H19": [], "TB_V22": [], "TB_V37": [], "TB_H37": [], "SIT": [], "X_SIT": [], "Y_SIT": []
  }
  cumulative_distance = 0
  seg = 1
  save_SIT=[]
  for idx in range(len(df)):
      
      df_temp = pd.DataFrame({
          "SIT": [],
          "X_SIT": [],
          "Y_SIT": [],
          "TB_V19": [],
          "TB_H19": [],
          "TB_V22": [],
          "TB_V37": [],
          "TB_H37": []
        })
      
      row = df.iloc[idx]

      for i in temp:
          temp[i].append(row[i])
      
      cumulative_distance += distances[idx]

      #print(cumulative_distance)
      #save_SIT.append(row["SIT"])
    
      if (idx + 1) > (df.shape[0]-1):
          next_dist = 0                   # Om sista tal, nästa distans är 0
      else:
          next_dist = distances[idx+1]    # Nästa distans
      
      if (cumulative_distance + next_dist) >= distance_segment:
          #print(f"Segment({seg}) is: {cumulative_distance * 0.001:.2f} [km] and contains {len(temp['SIT'])} values")
          
          #dist = np.linspace(0, cumulative_distance, num=len(temp["SIT"]))
          #plt.plot(dist, save_SIT) 
          #plt.xlabel("Along-track distance [m]") 
          #plt.ylabel("Sea ice thickness [m]")
          #plt.show()
          #breakpoint()

          # Remove segments where more than x% of values are NaN
          if np.isnan(temp["SIT"]).sum() * (100 / len(temp["SIT"])) >= 50:
              for i in temp:
                  temp[i] = []
              cumulative_distance = 0
              continue

          # Average all values in each valid segment with np.nanmean
          for i in ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT"]:
              #result[i].append(np.nanmean(temp[i]))

              # All coordinates in this whole segment should have the averge value 
              mean = np.nanmean(temp[i])
              df_temp[i] = np.full(len(temp[i]), mean)
          df_temp["X_SIT"] = temp["X_SIT"]
          df_temp["Y_SIT"] = temp["Y_SIT"]

          df_seg = pd.concat([df_seg, df_temp], ignore_index=True)

          # Middle coordinate of the whole segment
          #x_mid = (temp["X_SIT"][-1] + temp["X_SIT"][0]) / 2
          #y_mid = (temp["Y_SIT"][-1] + temp["Y_SIT"][0]) / 2
          #result["X_SIT"].append(x_mid)
          #result["Y_SIT"].append(y_mid)

          for i in temp:
              temp[i] = []
          cumulative_distance = 0
          seg += 1

  # If there are values left after segmenting
  if temp["SIT"]: 
      #print(f"Segment({seg}) is: {cumulative_distance * 0.001:.2f} [km] and contains {len(temp['SIT'])} values")

      for i in ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT"]:
          #result[i].append(np.nanmean(temp[i]))
          mean = np.nanmean(temp[i])
          df_temp[i] = np.full(len(temp[i]), mean)
      df_temp["X_SIT"] = temp["X_SIT"]
      df_temp["Y_SIT"] = temp["Y_SIT"]

      df_seg = pd.concat([df_seg, df_temp], ignore_index=True)

      #x_mid = (temp["X_SIT"][-1] + temp["X_SIT"][0]) / 2
      #y_mid = (temp["Y_SIT"][-1] + temp["Y_SIT"][0]) / 2
      #result["X_SIT"].append(x_mid)
      #result["Y_SIT"].append(y_mid)    
  
  return df_seg
