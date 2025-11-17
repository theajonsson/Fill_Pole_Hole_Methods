"""
File:       cartoplot.py
Purpose:    Provide function for plotting polar maps

Function:   make_ax, multi_cartoplot

Other:      Based on Robbie Mallet's cartoplot.py (https://github.com/robbiemallett/custom_modules/blob/master/cartoplot.py)
            Modified by Thea Jonsson since 2025-08-20
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs



"""
Function:   cartoplot
Purpose:    Plots one polar plot using cartopy, can use multiple sets of coordinates and data

Input:      coords_1 (float)
            coords_2 (float)
            data (float)
            title (string): title above the plot
            cbar_label (string): title to colorbar
            save_name (string): the plot will not be showed only directly saved in the given name
Return:     N/A
"""
def cartoplot(coords_1, coords_2, data,
                title=[],
                figsize=[10,5],
                hem='n',
                land=True, ocean=False,
                bounding_lat=65,
                gridlines=True,
                cbar_label="", save_name="", dot_size=0.1):
    

    fig = plt.figure(figsize=figsize)
    
    if hem == 'n':
        proj = ccrs.NorthPolarStereo()
        maxlat=90
    elif hem =='s':
        proj = ccrs.SouthPolarStereo()
        maxlat=-90
    else:
        raise

    data_array = [coords_1, coords_2, data]

    ax = fig.add_subplot(1, 1, 1, projection=proj)


    if ocean:
        ax.add_feature(cartopy.feature.OCEAN,zorder=2)
    if land:
        ax.add_feature(cartopy.feature.LAND, edgecolor='black',zorder=1)

    ax.set_extent([-180, 180, maxlat, bounding_lat], ccrs.PlateCarree())

    if gridlines:
        ax.gridlines()

    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right",size="5%", pad=0.05, axes_class=plt.Axes)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    for i in range(len(data)):
        m = ax.scatter(coords_1[i], coords_2[i], c=data[i],
                        s = dot_size,
                        transform=ccrs.epsg('3408'),
                        zorder=100, cmap="viridis_r", vmin=0, vmax=5)
    #ax.set_title(title)

    cb = plt.colorbar(m, cax=ax_cb)
    cb.set_label(cbar_label)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)
        
    if  save_name:
        dir = str(Path(__file__).resolve().parent/"Results/")
        save_format = ".png"
        fig.savefig(os.path.join(dir,save_name+save_format), dpi=300, bbox_inches="tight")
    else:    
        plt.show()
              