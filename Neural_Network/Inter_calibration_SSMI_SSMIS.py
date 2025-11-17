# 2025-11-13


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from cartoplot import cartoplot



# Plots histogram and cartoplot of the difference in TB+ical (SSM/I - SSMIS) per frequency channel
if False:
    SSM_I = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/TD_SSM_I_2006_03.csv"))
    SSMIS = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/TD_SSMIS_2006_03.csv"))

    # SSMIS.shape > SSM/I.shape -> using nearest-neighbour approach
    tree = KDTree(np.c_[SSMIS["X_SIT"], SSMIS["Y_SIT"]])
    dist, idx = tree.query(np.c_[SSM_I["X_SIT"], SSM_I["Y_SIT"]])

    df_matched = pd.DataFrame({
        "x" : SSMIS["X_SIT"].values[idx],
        "y" : SSMIS["Y_SIT"].values[idx],
        "TB_V19" : SSMIS["TB_V19"].values[idx],
        "TB_H19" : SSMIS["TB_H19"].values[idx],
        "TB_V22" : SSMIS["TB_V22"].values[idx],
        "TB_V37" : SSMIS["TB_V37"].values[idx],
        "TB_H37" : SSMIS["TB_H37"].values[idx],
    }) 

    channel_groups = {
        "TB_19": ["TB_V19", "TB_H19"],   
        "TB_22": ["TB_V22"],             
        "TB_37": ["TB_V37", "TB_H37"]    
    }
    for group_name, group in channel_groups.items():

        diff_TB = SSM_I[group] - df_matched[group]

        for channel in group:
            save_name = f"diff_{channel}"
            cartoplot([df_matched["x"]], [df_matched["y"]], [diff_TB[channel]], cbar_label="Brightness temperature [K]", save_name=save_name)
            plt.close()

        fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6), constrained_layout=True)
        if len(group) == 1:
            axs = [axs]
        
        for ax, channel in zip(axs, group):
            counts, bins, bars = ax.hist(diff_TB[channel], bins=10)
            ax.bar_label(bars, fmt='%d')
            ax.set_xlabel(f"Difference for {channel} [K]")
            ax.set_ylabel("Frequency")
        dir = str(Path(__file__).resolve().parent/"Results/Inter_calibration/")
        name = f"{group_name}.png"
        plt.savefig(os.path.join(dir,name), dpi=300, bbox_inches="tight")
        plt.close()
       

# Plots histogram and cartoplot of the difference in predicted SIT (SSM/I - SSMIS) 
# Trained on a model with TB and SIT, from October 2011 to March 2012, with 1 hidden layer and 4 hidden nodes
# Using this model with TB+ical values from March 2006
if False:
    SSM_I = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/Validate_PredSIT_SSM_I.csv"))
    SSMIS = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/Validate_PredSIT_SSMIS.csv"))

    # SSMIS.shape > SSM/I.shape -> using nearest-neighbour approach
    diff_TB = SSM_I["PredSIT"] - SSMIS["PredSIT"]

    cartoplot([SSMIS["X"]], [SSMIS["Y"]], [diff_TB], cbar_label="Sea ice thickness [m]", save_name="PredSIT_carto")
    plt.close()

    counts, bins, bars = plt.hist(diff_TB, bins=10)
    plt.bar_label(bars, fmt='%d')
    plt.xlabel(f"Difference in SIT [m]")
    plt.xlim([-0.2, 0.2])
    plt.ylabel("Frequency")
    dir = str(Path(__file__).resolve().parent/"Results/Inter_calibration/")
    plt.savefig(os.path.join(dir,"PredSIT_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()
