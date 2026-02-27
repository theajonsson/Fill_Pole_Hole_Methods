"""
File:       Inter_calibration_SSMI_SSMIS.py
Purpose:    PLots histogram and cartoplot of the difference in predicted SIT between SSM/I and SSMIS

Function:   N/A

Other:      Created by Thea Jonsson 2025-11-13
"""


import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from cartoplot import cartoplot



# Plots histogram and cartoplot of the difference in predicted SIT (SSM/I - SSMIS)
# 1. Create trainingdata for both SSM/I and SSMIS respectively
# 2. Create NN model for each respectively trainingdata
# 3. Use NN_ValidateModel to create predicted SIT on a choosen area
# 4. Then this code can be runned

SSM_I = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/Validate_PredSIT_SSM_I.csv"))
SSMIS = pd.read_csv(str(Path(__file__).resolve().parent/"Results/Inter_calibration/Validate_PredSIT_SSMIS.csv"))

diff_TB = SSM_I["PredSIT"] - SSMIS["PredSIT"]

cartoplot([SSMIS["X"]], [SSMIS["Y"]], [diff_TB], cbar_label="Difference in sea ice thickness [m]\n(SSM/I - SSMIS)", save_name="PredSIT_carto")
plt.close()

counts, bins, bars = plt.hist(diff_TB, bins=10)
plt.bar_label(bars, fmt='%d')
plt.xlabel(f"Difference in sea ice thickness [m]\n(SSM/I - SSMIS)")
plt.xlim([-0.8, 0.8])
plt.ylabel("Frequency")
dir = str(Path(__file__).resolve().parent/"Results/Inter_calibration/")
plt.savefig(os.path.join(dir,"PredSIT_hist.png"), dpi=300, bbox_inches="tight")
plt.close()
