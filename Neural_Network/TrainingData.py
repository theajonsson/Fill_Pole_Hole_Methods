"""
File:       TrainingData_SSMIS_AlongTrack.py
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



""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
hemisphere = "n"
lat_level = 60
""" ========================================================================================== """



# Create SSMIS training data (.csv) used for the NN.py for ONE MONTH
if False:
  start_time = time.time() 

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSM_I = pd.DataFrame(columns=columns) 
  index = 0

  all_x_SIT = np.array([])
  all_y_SIT = np.array([])
  all_SIT = np.array([])

  all_TB_V19 = np.array([])
  all_TB_H19 = np.array([])
  all_TB_V22 = np.array([])
  all_TB_V37 = np.array([])
  all_TB_H37 = np.array([])

  folder_path_SIT = str(Path(__file__).resolve().parent.parent/"Data/Envisat_SatSwath/2006/03") # Investigate 2007
  folder_path_SSM_I = str(Path(__file__).resolve().parent.parent/"Data/TB_2006_03_SSM_I")

  files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
  files_SSM_I = sorted([f for f in os.listdir(folder_path_SSM_I) if f[0].isalnum()])

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

  for file_SIT, files_SSM_I in zip(files_SIT, files_SSM_I):

    path_SIT = os.path.join(folder_path_SIT, file_SIT)
    path_SSM_I = os.path.join(folder_path_SSM_I, files_SSM_I)
    
    x_SIT, y_SIT, SIT = fd.format_SIT(path_SIT)

    all_x_SIT = np.append(all_x_SIT, x_SIT)
    all_y_SIT = np.append(all_y_SIT, y_SIT)
    all_SIT = np.append(all_SIT, SIT)

    index = 0
    vh = [0, 1, 2, 3, 4]
    for j in range(len(vh)):
      x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, near_TB, nearest_TB_coords = fd.format_SSM_I(x_SIT, y_SIT, path_SSM_I, group_SSM_I, vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

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
    print(file_SIT)
    print(files_SSM_I)
    
  df_TB_SSM_I["SIT"] = all_SIT
  df_TB_SSM_I["X_SIT"] = all_x_SIT
  df_TB_SSM_I["Y_SIT"] = all_y_SIT

  df_TB_SSM_I["TB_V19"] = all_TB_V19 
  df_TB_SSM_I["TB_H19"] = all_TB_H19 
  df_TB_SSM_I["TB_V22"] = all_TB_V22 
  df_TB_SSM_I["TB_V37"] = all_TB_V37 
  df_TB_SSM_I["TB_H37"] = all_TB_H37 

  df_TB_SSM_I = df_TB_SSM_I.dropna(subset=["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"])

  df_TB_SSM_I.to_csv(str(Path(__file__).resolve().parent.parent/"Data/NN/TD_SSM_I_2006_03.csv"), index=False)
  print(df_TB_SSM_I.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")



# Create SSMIS training data (.csv) used for the NN.py for ONE MONTH
if False:
  start_time = time.time() 

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 
  index = 0

  all_x_SIT = np.array([])
  all_y_SIT = np.array([])
  all_SIT = np.array([])

  all_TB_V19 = np.array([])
  all_TB_H19 = np.array([])
  all_TB_V22 = np.array([])
  all_TB_V37 = np.array([])
  all_TB_H37 = np.array([])

  folder_path_SIT = str(Path(__file__).resolve().parent.parent/"Data/Envisat_SatSwath/2006/03") # Investigate 2007
  folder_path_SSMIS = str(Path(__file__).resolve().parent.parent/"Data/TB_2006_03_SSMIS")

  files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
  files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

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
    print(file_SIT)
    print(file_SSMIS)
    
  df_TB_SSMIS["SIT"] = all_SIT
  df_TB_SSMIS["X_SIT"] = all_x_SIT
  df_TB_SSMIS["Y_SIT"] = all_y_SIT

  df_TB_SSMIS["TB_V19"] = all_TB_V19 
  df_TB_SSMIS["TB_H19"] = all_TB_H19 
  df_TB_SSMIS["TB_V22"] = all_TB_V22 
  df_TB_SSMIS["TB_V37"] = all_TB_V37 
  df_TB_SSMIS["TB_H37"] = all_TB_H37 

  df_TB_SSMIS = df_TB_SSMIS.dropna(subset=["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"])

  df_TB_SSMIS.to_csv(str(Path(__file__).resolve().parent.parent/"Data/NN/TD_SSMIS_2006_03.csv"), index=False)
  print(df_TB_SSMIS.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")

# Plot TB per channel (y-axis) against SIT (x-axis)
if False:
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

        ax.set_title(f"{channel} vs SIT")
        ax.set_xlabel("SIT [m]")
        ax.set_ylabel(f"{channel} [K]")
        ax.legend(loc="lower right")
        ax.grid(True)

    name = f"/Users/theajonsson/Desktop/TBvsSIT_{channel}.png"
    plt.savefig(name, dpi=300, bbox_inches="tight")

# Plot histogram of SIT and TB per channel
if False:
  plt.hist(df_TB_SSMIS["SIT"])
  plt.savefig("/Users/theajonsson/Desktop/Hist_SIT.png", dpi=300, bbox_inches="tight")
