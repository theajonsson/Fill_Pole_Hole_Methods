# 2025-11-13

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import linregress
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable





def read_file(file_path):
    dates = []
    values = []
    with open(file_path, "r") as f:
        for line in f:
            date, value = line.strip().split(":")
            dates.append(date)
            values.append(float(value))
    return dates, np.array(values)

def split_by_season(dates, values):
    split_dates = []
    split_values = []
    temp_dates = []
    temp_values = []
    
    for i in range(len(dates)):
        month = dates[i].split('-')[1]  # Extract the month from the date
        
        # If month is October (start of a new period), plot the previous data
        if month == "10" and temp_dates:
            split_dates.append(temp_dates)
            split_values.append(temp_values)
            
            temp_dates = []
            temp_values = []
        
        # Convert the date string to datetime
        temp_dates.append(datetime.strptime(dates[i], '%Y-%m'))
        temp_values.append(values[i])

    if temp_dates:
        split_dates.append(temp_dates)
        split_values.append(temp_values)

    return split_dates, split_values



def split_by_month(dates, values):
    month_groups = {m: {"dates": [], "values": []} for m in ["10","11","12","01","02","03","04"]}

    for date_str, value in zip(dates, values):
        month = date_str.split('-')[1] 

        if month in month_groups:
            month_groups[month]["dates"].append(datetime.strptime(date_str, "%Y-%m"))
            month_groups[month]["values"].append(value)

    return month_groups

def calculate_slopes(month_groups):
    slopes = {}
        
    for month, data in month_groups.items():
        dates = data["dates"]
        values = data["values"]

        if len(dates) < 2:
            slopes[month] = None
            continue

        decimal_years = [
            d.year + (d.timetuple().tm_yday - 1) / 365.0
            for d in dates
        ]

        result = linregress(decimal_years, values)
        slopes[month] = result.slope

    return slopes

def plot_months(month_groups):
    month_names = {
    "10": "Oct",
    "11": "Nov",
    "12": "Dec",
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for month, data in month_groups.items():
        if not data["dates"]:
            continue

        x = data["dates"]
        y = data["values"]

        ax.scatter(x, y)

        x_ord = [d.toordinal() for d in x]
        slope_day, intercept, r, p, stderr = linregress(x_ord, y)

        slope_year = slope_day * 365.25

        x_fit = sorted(x)
        x_fit_ord = [d.toordinal() for d in x_fit]
        y_fit = [slope_day * xo + intercept for xo in x_fit_ord]

        ax.plot(x_fit, y_fit, linewidth=2, label=f"{month_names.get(month, month)}\nSlope = {slope_year:.2f} km$^3$/year", alpha=0.5)
    
    plt.xlim([datetime(2002, 1, 1), datetime(2012, 12, 31)])
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())             # Yearly thicks, diplayed in the beginning of each year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))     # Display text of year 
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())            # Monthly ticks 
    plt.gca().tick_params(which="minor", axis="x", length=5, width=0.5, color="gray")
    plt.xticks(rotation=45)
    ax.set_ylim(0, 20000) 
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea ice volume [km$^3$]")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()



halo_file = str(Path(__file__).resolve().parent/"HaloMethod_EnvPeriod.txt")
type_file = str(Path(__file__).resolve().parent/"IceAgeMethod_EnvPeriod.txt")
nn_file = str(Path(__file__).resolve().parent/"NNMethod_EnvPeriod.txt")



# CREATE PLOT FOR ESTIMATED SIV FOR ENVISTA PERIOD (2002-2012)
if False:
    halo_dates, halo_values = read_file(halo_file)
    type_dates, type_values = read_file(type_file)
    nn_dates, nn_values = read_file(nn_file)

    halo_split_dates, halo_split_values = split_by_season(halo_dates, halo_values)
    type_split_dates, type_split_values = split_by_season(type_dates, type_values)
    nn_split_dates, nn_split_values = split_by_season(nn_dates, nn_values)


    plt.figure(figsize=(10, 6))

    for i in range(len(halo_split_dates)):
        plt.plot(halo_split_dates[i], halo_split_values[i], label="Halo method" if i == 0 else "", 
                 color="#276CF5", marker="o", markersize=6.5, linestyle="-", linewidth=3.5)

    for i in range(len(type_split_dates)):
        plt.plot(type_split_dates[i], type_split_values[i], label="Ice age method" if i == 0 else "", 
                 color="#F54927",marker="v", markersize=5,linestyle="--", linewidth=2)
    
    for i in range(len(nn_split_dates)):
        plt.plot(nn_split_dates[i], nn_split_values[i], label="Neural network method" if i == 0 else "", 
                 color="#27F5B0", marker="p", markersize=5, linestyle="-.")

    plt.xlabel("Year")
    plt.ylabel("Sea ice volume [km$^3$]")
    plt.xlim([datetime(2002, 1, 1), datetime(2012, 12, 31)])
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())             # Yearly thicks, diplayed in the beginning of each year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))     # Display text of year 
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())            # Monthly ticks 
    plt.gca().tick_params(which="minor", axis="x", length=5, width=0.5, color="gray")
    plt.ylim(0,20000)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    dir = str(Path(__file__).resolve().parent/"Results/")
    plt.savefig(os.path.join(dir,"EstimatingSIV_EnvPeriod.png"), dpi=300, bbox_inches="tight")
    plt.show()



# CREATE PLOT FOR THE SLOPE OF SIV IN EACH MONTHS 
if True:
    halo_dates, halo_values = read_file(halo_file)
    type_dates, type_values = read_file(type_file)
    nn_dates, nn_values = read_file(nn_file)

    halo_month = split_by_month(halo_dates, halo_values)
    type_month = split_by_month(type_dates, type_values)
    nn_month = split_by_month(nn_dates, nn_values)

    breakpoint()

    

    breakpoint()

    halo_slopes = calculate_slopes(halo_month)
    type_slopes = calculate_slopes(type_month)
    nn_slopes = calculate_slopes(nn_month)

    plot_months(halo_month)
    plot_months(type_month)
    plot_months(nn_month)

    df = pd.DataFrame({
        "Month": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"],
        "Halo": list(halo_slopes.values()),
        "Ice age": list(type_slopes.values()),
        "Neural network": list(nn_slopes.values())
    })

    halo_avg = df["Halo"].mean()
    type_avg = df["Ice age"].mean()
    nn_avg = df["Neural network"].mean()
    df.loc["Average"] = ["Average", halo_avg, type_avg, nn_avg]

    pivot_data = df.set_index("Month").transpose()

    fig, ax = plt.subplots(figsize=(7, 3))
    hm = sns.heatmap(pivot_data, annot=True, cmap="viridis", fmt=".1f", linewidths=0.5, linecolor="white", 
                cbar=False, vmin=-300, vmax=-40, ax=ax) #cbar_kws={"label": "Slope [km$^3$/year]", "orientation": "horizontal", "pad": 0.2})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.4)
    cbar = plt.colorbar(hm.collections[0], cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")
    cax.xaxis.set_label_position("top")
    for spine in cax.spines.values():
        spine.set_visible(False)
    cbar.set_label("Slope [km$^3$/year]")
    
    ax.set_ylabel("Method", fontsize=10)
    ax.set_xlabel("")
    plt.tight_layout()
    output_dir = str(Path(__file__).resolve().parent/"Results/")
    plt.savefig(os.path.join(output_dir, "Slope_Heatmap_EnvPeriod.png"), dpi=300, bbox_inches="tight")
    plt.show()
