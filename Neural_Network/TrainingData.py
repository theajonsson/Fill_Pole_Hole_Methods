"""
File:       TrainingData.py
Purpose:    Create training data (.csv) consisting of brightness temperature (TB) [K] and 
            along-track sea ice thickness (SIT) [m], this can then be used to create a model in NN.py

Needs:      cartoplot.py, format_data.py

Other:      Created by Thea Jonsson 2025-09-19
"""

import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import format_data as fd 
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from cartoplot import cartoplot



""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
hemisphere = "n"
lat_level = 60
""" ========================================================================================== """


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

  breakpoint()

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

# Original
if False:
  start_time = time.time() 

  """
  years_months = {
    "2006" : ["10", "11", "12"],
    "2007" : ["01", "02", "03", "04"]
  }"""
  years_months={"2006":["10"]}

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

  for year, months in years_months.items():
    for month in months:
      print(f"\n----- Processing {year}-{month} -----")

      folder_path_SIT = str(Path(__file__).resolve().parent.parent/f"Data/Envisat_SatSwath/{year}/{month}")
      folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMIS/{year}/{month}")

      files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
      files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

      all_x_SIT = np.array([])
      all_y_SIT = np.array([])
      all_SIT = np.array([])

      all_TB_V19 = np.array([])
      all_TB_H19 = np.array([])
      all_TB_V22 = np.array([])
      all_TB_V37 = np.array([])
      all_TB_H37 = np.array([])

      for file_SIT, file_SSMIS in zip(files_SIT, files_SSMIS):

        path_SIT = os.path.join(folder_path_SIT, file_SIT)
        path_SSMIS = os.path.join(folder_path_SSMIS, file_SSMIS)
        
        x_SIT, y_SIT, SIT = fd.format_SIT(path_SIT)
        all_x_SIT = np.append(all_x_SIT, x_SIT)
        all_y_SIT = np.append(all_y_SIT, y_SIT)
        all_SIT = np.append(all_SIT, SIT)

        index = 0
        for i in range(len(group_SSMIS)):
          if i == 0:
              vh = [1, 0, 2]
          else:
              vh = [1, 0]
          for j in range(len(vh)):
            x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, near_TB, nearest_TB_coords = fd.format_SSMIS(x_SIT, y_SIT, path_SSMIS, group_SSMIS[i], vh[j], lons_valid, lats_valid, land_mask_data, debug=False)
            
            if index == 0:
              all_TB_V19 = np.append(all_TB_V19, near_TB)
            elif index == 1:
              all_TB_H19 = np.append(all_TB_H19, near_TB)
            elif index == 2:
              all_TB_V22 = np.append(all_TB_V22, near_TB)
            elif index == 3:
              all_TB_V37 = np.append(all_TB_V37, near_TB)
            else:
              all_TB_H37 = np.append(all_TB_H37, near_TB)  

            index += 1
        print(f"Processed file: {file_SIT}, {file_SSMIS}")
        break
      df_month = pd.DataFrame({
              "SIT": all_SIT,
              "X_SIT": all_x_SIT,
              "Y_SIT": all_y_SIT,
              "TB_V19": all_TB_V19,
              "TB_H19": all_TB_H19,
              "TB_V22": all_TB_V22,
              "TB_V37": all_TB_V37,
              "TB_H37": all_TB_H37
          })

      df_TB_SSMIS = pd.concat([df_TB_SSMIS, df_month], ignore_index=True)

  df_TB_SSMIS = df_TB_SSMIS.dropna(subset=["TB_V19","TB_H19","TB_V22","TB_V37","TB_H37"])

  breakpoint()

  #df_TB_SSMIS.to_csv(str(Path(__file__).resolve().parent/"Results/TrainingData/TD_2006_2007.csv"), index=False)
  df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TD_2006_2007.csv", index=False)
  print(df_TB_SSMIS.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")

# Split tracks
if True:
  start_time = time.time() 

  years_months = {
    "2006" : ["10", "11", "12"],
    "2007" : ["01", "02", "03", "04"]
  }

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

  for year, months in years_months.items():
    for month in months:
      print(f"\n----- Processing {year}-{month} -----")

      folder_path_SIT = str(Path(__file__).resolve().parent.parent/f"Data/Envisat_SatSwath/{year}/{month}")
      folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMIS/{year}/{month}")

      files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
      files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

      all_x_SIT = np.array([])
      all_y_SIT = np.array([])
      all_SIT = np.array([])

      all_TB_V19 = np.array([])
      all_TB_H19 = np.array([])
      all_TB_V22 = np.array([])
      all_TB_V37 = np.array([])
      all_TB_H37 = np.array([])

      for file_SIT, file_SSMIS in zip(files_SIT, files_SSMIS):

        path_SIT = os.path.join(folder_path_SIT, file_SIT)
        path_SSMIS = os.path.join(folder_path_SSMIS, file_SSMIS)
        
        x_SIT, y_SIT, SIT = fd.format_SIT(path_SIT)
        all_x_SIT = np.append(all_x_SIT, x_SIT)
        all_y_SIT = np.append(all_y_SIT, y_SIT)
        all_SIT = np.append(all_SIT, SIT)

        index = 0
        tb_order = [
            ("scene_env1", 1),   # 19V
            ("scene_env1", 0),   # 19H
            ("scene_env1", 2),   # 22V
            ("scene_env2", 1),   # 37V
            ("scene_env2", 0)    # 37H
        ]
        for group, channel in tb_order:
          print(f"{group}   {channel}")
          
          x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, near_TB, nearest_TB_coords = fd.format_SSMIS(x_SIT, y_SIT, path_SSMIS, group, channel, lons_valid, lats_valid, land_mask_data, debug=False)
          
          if index == 0:
            all_TB_V19 = np.append(all_TB_V19, near_TB)
          elif index == 1:
            all_TB_H19 = np.append(all_TB_H19, near_TB)
          elif index == 2:
            all_TB_V22 = np.append(all_TB_V22, near_TB)
          elif index == 3:
            all_TB_V37 = np.append(all_TB_V37, near_TB)
          else:
            all_TB_H37 = np.append(all_TB_H37, near_TB)  
        
          index += 1
          
        print("Lengths:",len(all_SIT), len(all_x_SIT), len(all_y_SIT),len(all_TB_V19), len(all_TB_H19),len(all_TB_V22), len(all_TB_V37), len(all_TB_H37))
          
        df_file = pd.DataFrame({
          "SIT": all_SIT,
          "X_SIT": all_x_SIT,
          "Y_SIT": all_y_SIT,
          "TB_V19": all_TB_V19,
          "TB_H19": all_TB_H19,
          "TB_V22": all_TB_V22,
          "TB_V37": all_TB_V37,
          "TB_H37": all_TB_H37
        })
      
        df_file_seg = split_tracks(df_file, distance_segment=80000)
        print(f"Processed file: {file_SIT}, {file_SSMIS}")

      df_TB_SSMIS = pd.concat([df_TB_SSMIS, df_file_seg], ignore_index=True)

  print(np.isnan(df_TB_SSMIS).sum())    
  df_TB_SSMIS = df_TB_SSMIS.dropna() #subset=["TB_V19","TB_H19","TB_V22","TB_V37","TB_H37"]

  df_TB_SSMIS.to_csv(str(Path(__file__).resolve().parent/"Results/TrainingData/TD_2006_2007.csv"), index=False)
  print(df_TB_SSMIS.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")



#df_TB_SSMIS = pd.read_csv("/Users/theajonsson/Desktop/TD_2006_2007.csv")

#cols = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]
#df_TB_SSMIS = df_TB_SSMIS[df_TB_SSMIS[cols].ge(5).all(axis=1)]


# Plot TB per channel (y-axis) against SIT (x-axis)
if True:
   channel_groups = [
      ["TB_V19", "TB_H19"],   
      ["TB_V22"],             
      ["TB_V37", "TB_H37"]    
    ]
   for group in channel_groups:
    fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6), constrained_layout=True)

    if len(group) == 1:
       axs = [axs]

    for ax, channel in zip(axs, group):
        
        bias = np.mean(df_TB_SSMIS[channel] - df_TB_SSMIS["SIT"])

        slope, intercept, r_value, p_value, std_err = linregress(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel])
        r_squared = r_value ** 2

        rmse = mean_squared_error(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], squared=False)

        hb = ax.hexbin(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], gridsize=50, mincnt=6, label=f"Bias: {bias:.3f} \nR-squared: {r_squared:.3f} \nRMSE={rmse:.3f}")
        ax.plot(df_TB_SSMIS["SIT"], intercept + slope * df_TB_SSMIS["SIT"], color="red",
                label=f"Fitted line")

        #ax.set_title(f"{channel} vs SIT")
        ax.set_xlabel("Envisat sea ice thickness [m]")
        ax.set_ylabel(f"{channel} [K]")
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.set_xlim(0,10)
        ax.set_ylim(0,350)

    #dir = str(Path(__file__).resolve().parent/"Results/TrainingData/")
    dir = "/Users/theajonsson/Desktop/"
    name = f"TBvsSIT_{channel}.png"
    plt.savefig(os.path.join(dir,name), dpi=300, bbox_inches="tight")
    plt.close()

# Plot histogram of SIT and TB per channel
if True:
  counts, bin_edges, patches = plt.hist(df_TB_SSMIS["SIT"], bins=10)
  plt.bar_label(patches, fontsize=8)
  plt.xlabel("Sea ice thickness [m]")
  plt.ylabel("Frequency")

  #dir = str(Path(__file__).resolve().parent/"Results/TrainingData/")
  name = "Hist_SIT.png"
  plt.savefig(os.path.join(dir,name), dpi=300, bbox_inches="tight")
  plt.close()





# Create SSM/I training data (.csv) used for the NN.py for ONE MONTH
if False:
  start_time = time.time() 

  """
  years_months = {
    "2006" : ["10", "11", "12"],
    "2007" : ["01", "02", "03", "04"]
  }"""
  years_months={"2006":["10"]}

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMI = pd.DataFrame(columns=columns) 

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

  for year, months in years_months.items():
    for month in months:
      print(f"\n----- Processing {year}-{month} -----")

      folder_path_SIT = str(Path(__file__).resolve().parent.parent/f"Data/Envisat_SatSwath/{year}/{month}")
      folder_path_SSMI = str(Path(__file__).resolve().parent.parent/f"Data/TB_SSMI/{year}/{month}")

      files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
      files_SSMI = sorted([f for f in os.listdir(folder_path_SSMI) if f[0].isalnum()])

      all_x_SIT = np.array([])
      all_y_SIT = np.array([])
      all_SIT = np.array([])

      all_TB_V19 = np.array([])
      all_TB_H19 = np.array([])
      all_TB_V22 = np.array([])
      all_TB_V37 = np.array([])
      all_TB_H37 = np.array([])

      for file_SIT, file_SSMI in zip(files_SIT, files_SSMI):

        path_SIT = os.path.join(folder_path_SIT, file_SIT)
        path_SSMI = os.path.join(folder_path_SSMI, file_SSMI)
        
        x_SIT, y_SIT, SIT = fd.format_SIT(path_SIT)
        all_x_SIT = np.append(all_x_SIT, x_SIT)
        all_y_SIT = np.append(all_y_SIT, y_SIT)
        all_SIT = np.append(all_SIT, SIT)

        index = 0
        vh = [0, 1, 2, 3, 4]
        for j in range(len(vh)):
          x_TB_SSMI, y_TB_SSMI, TB_SSMI, near_TB, nearest_TB_coords = fd.format_SSM_I(x_SIT, y_SIT, path_SSMI, group_SSM_I, vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

          if index == 0:
            all_TB_V19 = np.append(all_TB_V19, near_TB)
          elif index == 1:
            all_TB_H19 = np.append(all_TB_H19, near_TB)
          elif index == 2:
            all_TB_V22 = np.append(all_TB_V22, near_TB)
          elif index == 3:
            all_TB_V37 = np.append(all_TB_V37, near_TB)
          else:
            all_TB_H37 = np.append(all_TB_H37, near_TB)  

          index += 1
          
        print("Lengths:",len(all_SIT), len(all_x_SIT), len(all_y_SIT),len(all_TB_V19), len(all_TB_H19),len(all_TB_V22), len(all_TB_V37), len(all_TB_H37))
          
        df_file = pd.DataFrame({
          "SIT": all_SIT,
          "X_SIT": all_x_SIT,
          "Y_SIT": all_y_SIT,
          "TB_V19": all_TB_V19,
          "TB_H19": all_TB_H19,
          "TB_V22": all_TB_V22,
          "TB_V37": all_TB_V37,
          "TB_H37": all_TB_H37
        })
      
        df_file_seg = split_tracks(df_file, distance_segment=50000)
        print(f"Processed file: {file_SIT}, {file_SSMI}")

      df_TB_SSMI = pd.concat([df_TB_SSMI, df_file_seg], ignore_index=True)
      
  df_TB_SSMI = df_TB_SSMI.dropna(subset=["TB_V19","TB_H19","TB_V22","TB_V37","TB_H37"])

  #df_TB_SSMI.to_csv(str(Path(__file__).resolve().parent/"Results/TrainingData/TD_2006_2007.csv"), index=False)
  df_TB_SSMI.to_csv("/Users/theajonsson/Desktop/TD_SSMI_2006_03.csv", index=False)
  print(df_TB_SSMI.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")
  