import pandas as pd
import geopandas as gpd
import glob
import os
from shapely import wkt

# from optimization_parameters import *
from variable_definitions import *
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import pd2gpd
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

path = "sensitivity_analyses/30perc/"

list_of_paths = [
    "20220120-113201_optimization_result_charging_stations.csv",
    "20220120-114031_optimization_result_charging_stations.csv",
    "20220120-115211_optimization_result_charging_stations.csv",
    "20220120-120527_optimization_result_charging_stations.csv",
    "20220120-122052_optimization_result_charging_stations.csv",
    "20220120-123818_optimization_result_charging_stations.csv",
    "20220120-125817_optimization_result_charging_stations.csv",
    "20220120-132155_optimization_result_charging_stations.csv",
    "20220120-134652_optimization_result_charging_stations.csv",
    "20220120-140916_optimization_result_charging_stations.csv",
    "20220120-143244_optimization_result_charging_stations.csv",
    "20220120-145619_optimization_result_charging_stations.csv",
    "20220120-151943_optimization_result_charging_stations.csv",
]

results = []

range_max = 1400
range_min = 200
step_size = 100
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path + list_of_paths[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []

for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path + list_of_paths[ij])
    df[ranges[ij]] = temp_df.pYi_dir
    nb_x.append(temp_df.pXi.sum())
nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

# plot

fig, ax = plt.subplots(figsize=(8, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 14
ax2 = ax.twinx()
plt.xlim([range_min - step_size, range_max + step_size])
ax.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#457b9d"

df_to_plot.boxplot(
    ax=ax,
    widths=(50),
    notch=True,
    patch_artist=True,
    boxprops=dict(facecolor=c, color=c),
    capprops={"color": c, "linewidth": 2},
    whiskerprops={"color": c, "linewidth": 2},
    flierprops={"color": c, "markeredgewidth": 2, "markeredgecolor": c},
    medianprops=dict(color=c),
    positions=ranges,
    labels=ranges,
)
ax.set_xlabel("driving range (km)")
plt.grid("off")
ax.set_ylabel("# charging poles per charging point", color="#1d3557", fontsize=10)
ax2.set_ylabel(
    "# charging points", color="#723d46", rotation=-90, labelpad=12, fontsize=12
)
ax2.grid(False)
l0 = ax2.plot(
    list(ranges) + [range_max + step_size],
    nb_x,
    color="#723d46",
    linewidth=4,
    label="# charging points",
)
ax2.set_ylim([36, 55])
ax.set_ylim([0, 85])
l1 = ax.plot(
    [range_min - step_size, range_max + step_size],
    [10000 / 150] * len([range_min - step_size, range_max + step_size]),
    "--",
    color="grey",
    linewidth=2,
    label="Max. # of charging poles at a charging point",
)
ax.tick_params(axis="y", colors="#1d3557", labelsize=10)
ax2.tick_params(axis="y", colors="#723d46", labelsize=10)

ax2.spines["left"].set_color("#1d3557")
ax.spines["left"].set_color("#1d3557")  # setting up Y-axis tick color to red
ax2.spines["right"].set_color("#723d46")
ax2.spines["top"].set_visible(False)
ax.spines["top"].set_visible(False)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

lns = l0 + l1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=12)
# plt.show()
plt.savefig(path + "range_30.pdf", bbox_inches="tight")
