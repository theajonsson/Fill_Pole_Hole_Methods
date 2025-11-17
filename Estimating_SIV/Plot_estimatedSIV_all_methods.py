# 2025-11-13

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

# Plot estimated SIV for the whole Envisat period
halo = str(Path(__file__).resolve().parent/"AveragingSIT_method.txt")
type = str(Path(__file__).resolve().parent/"AveragingType_method.txt")
NN = str(Path(__file__).resolve().parent/"NN_method.txt")

breakpoint()

dates, values = [], []

with open(type, "r") as f:
    for line in f:
        key, val = line.strip().split(":")
        dates.append(key.strip())
        values.append(float(val.strip()))


