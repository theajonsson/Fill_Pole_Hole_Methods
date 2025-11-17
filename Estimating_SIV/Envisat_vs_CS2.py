"""
File:       Envisar_vs_CS2.py
Purpose:    Create plots for overlap timeperiod for EnviSat and CryoSat-2,
            by plotting theirs respective sea ice thickness (SIT) [m]

Function:   env_vs_cs2

Other:      Created by Thea Jonsson 2025-10-16
"""

import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.odr import ODR, Model, RealData
import os
from PIL import Image



"""
Function:   env_vs_cs2
Purpose:    Create individual plots of EnvSat SITs against CS-2 SITs

Input:      year, month (string)
Return:     N/A
"""
def env_vs_cs2(year, month):

    file_env = os.path.join(path_env, year, f"ESACCI-SEAICE-L3C-SITHICK-RA2_ENVISAT-NH25KMEASE2-{year}{month}-fv2.0.nc")
    file_cs2 = os.path.join(path_cs2, year, f"ESACCI-SEAICE-L3C-SITHICK-SIRAL_CRYOSAT2-NH25KMEASE2-{year}{month}-fv2.0.nc")

    with nc.Dataset(file_env, "r") as dataset_env:
        SIT_env = dataset_env["sea_ice_thickness"][:].filled(np.nan).flatten()

    with nc.Dataset(file_cs2, "r") as dataset_cs2:
        SIT_cs2 = dataset_cs2["sea_ice_thickness"][:].filled(np.nan).flatten()

    data = np.stack((SIT_env, SIT_cs2), axis=1)
    valid_mask = np.all(np.isfinite(data), axis=1)
    SIT_env = SIT_env[valid_mask]
    SIT_cs2 = SIT_cs2[valid_mask]

    def linear_model(B, x):
        return B[0] + B[1] * x
    #
    model = Model(linear_model)
    real_data = RealData(SIT_cs2, SIT_env)
    odr = ODR(real_data, model, beta0=[0., 1.])
    out = odr.run()
    #
    intercept = out.beta[0]
    slope = out.beta[1]
    #
    #y_pred = intercept + slope * SIT_cs2
    #ss_res = np.sum((SIT_env - y_pred)**2)
    #ss_tot = np.sum((SIT_env - np.mean(SIT_env))**2)
    #r_squared = 1 - ss_res/ss_tot

    bias = np.mean(SIT_env - SIT_cs2)

    #slope, intercept, r_value, p_value, std_err = linregress(SIT_cs2, SIT_env)
    #r_squared = r_value ** 2

    rmse = mean_squared_error(SIT_cs2, SIT_env, squared=False)

    mean_cs2 = np.mean(SIT_cs2)
    mean_env = np.mean(SIT_env)
    min_val = min(SIT_cs2.min(), SIT_env.min())
    max_val = max(SIT_cs2.max(), SIT_env.max())

    plt.scatter(SIT_cs2, SIT_env, s=10, color="#43a2ca", alpha=0.5, label=f"RMSE={rmse:.3f} \nBias: {bias:.3f}")# \nR-squared: {r_squared:.3f}")
    plt.scatter(mean_cs2, mean_env, color="#2c48b5", s=50, marker="x", zorder=10, label="Center of mass")
    
    #plt.plot(SIT_cs2, intercept + slope * SIT_cs2, color="#8856a7", label="Fitted line")
    plt.plot(SIT_cs2, intercept + slope * SIT_cs2, color="#8856a7", label="ODR fit")
    
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", label="Optimal line")
    plt.xlabel("CS-2 SIT [m]")
    plt.ylabel("Envisat SIT [m]")
    plt.title(f"{year}-{month}")
    plt.legend(loc="upper right")
    plt.grid(True)

    dir = str(Path(__file__).resolve().parent/"Results/Envisat_against_CS2/")
    name = f"{year}_{month}"
    plt.savefig(os.path.join(dir,name+".png"), dpi=300, bbox_inches="tight") 
    plt.close()

    print(f"Saved plot for {year}-{month}")





path_env = str(Path(__file__).resolve().parent.parent/"Data/Envisat_Monthly/")
path_cs2 = str(Path(__file__).resolve().parent.parent/"Data/Cryosat_Monthly/")

data = {"2010": ["11", "12"],
        "2011": ["01", "02", "03", "04", "10", "11", "12"],
        "2012": ["01", "02", "03"]}

# Run the data
if False:
    for year, months in data.items():
        for month in months:
            env_vs_cs2(year, month)





# Plot all saved indivudal plots as two plots (2010/2011 and 2011/2012)
if False:
    dir = str(Path(__file__).resolve().parent/"Results/Envisat_against_CS2/")

    months_1011 = ["2010_11.png", "2010_12.png", "2011_01.png", "2011_02.png", "2011_03.png", "2011_04.png"]
    images_1011 = [Image.open(os.path.join(dir, f)) for f in months_1011]

    fig1, axs1 = plt.subplots(nrows=3, ncols=2, figsize=(8.27, 11.69))

    for i, ax in enumerate(axs1.flat):
        ax.imshow(images_1011[i])
        ax.axis("off")

    fig1.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(dir,"Plot_2010_2011.png"), dpi=300, bbox_inches="tight")
    plt.close()



    months_1112 = ["2011_10.png", "2011_11.png", "2011_12.png", "2012_01.png", "2012_02.png", "2012_03.png"]
    images_1112 = [Image.open(os.path.join(dir, f)) for f in months_1112]

    fig2, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(8.27, 11.69))

    for i, ax in enumerate(axs2.flat):
        ax.imshow(images_1112[i])
        ax.axis("off")

    fig2.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(dir,"Plot_2011_2012.png"), dpi=300, bbox_inches="tight")
    plt.close()
