import pandas as pd
import geopandas as gpd
import glob
import os
from shapely import wkt
import matplotlib.patches as mpatches
# from optimization_parameters import *
from variable_definitions import *
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import pd2gpd
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

path = "sensitivity_analyses/TC range/"

list_of_paths = [
    "20220201-133851_TF200_optimization_result_charging_stations.csv",
    "20220201-134907_TF300_optimization_result_charging_stations.csv",
    "20220201-140143_TF400_optimization_result_charging_stations.csv",
    "20220201-142101_TF500_optimization_result_charging_stations.csv",
    "20220201-144117_TF600_optimization_result_charging_stations.csv",
    "20220201-150353_TF700_optimization_result_charging_stations.csv",
    "20220201-153742_TF800_optimization_result_charging_stations.csv",
    "20220201-160259_TF900_optimization_result_charging_stations.csv",
    "20220201-162724_TF1000_optimization_result_charging_stations.csv",
    "20220201-165552_TF1100_optimization_result_charging_stations.csv",
    "20220201-172552_TF1200_optimization_result_charging_stations.csv",
    "20220201-175253_TF1300_optimization_result_charging_stations.csv",
    "20220201-183505_TF1400_optimization_result_charging_stations.csv"
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
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

# plot

fig, ax = plt.subplots(figsize=(8, 4))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12

font = {'family': "Franklin Gothic Book", 'fontsize':12
        }
ax2 = ax.twinx()
plt.xlim([range_min - step_size, range_max + step_size])
ax.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#6096ba"
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12

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
ax.set_xlabel("driving range (km)", fontname="Franklin Gothic Book")
ax.set_yticklabels(labels= list(range(0, 55,5)),fontname="Franklin Gothic Book")
ax2.set_xticklabels(labels= list(range(200, 1500, 100)), fontname="Franklin Gothic Book")
ax.set_xticklabels(labels= list(range(200, 1500, 100)), fontname="Franklin Gothic Book")

plt.grid("off")
ax.set_ylabel("Nb. charging stations per charging point", color="#1d3557", fontdict=font)
ax2.set_ylabel(
    "Nb. charging points", color="#723d46", rotation=-90, labelpad=12, fontsize=12
)
ax2.grid(False)
l0 = ax2.plot(
    list(ranges),
    nb_x,
    marker="o",
    color="#723d46",
    linewidth=2,
    label="Nb. of CP",
)
ax2.set_ylim([36, 66])
ax.set_ylim([0, 50])
l1 = ax.plot(
    [range_min - step_size, range_max + step_size],
    [int(12000 / 350)] * len([range_min - step_size, range_max + step_size]),
    "--",
    color="grey",
    linewidth=2,
    label="Max. nb. of CS at a CP",
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
ax.annotate('', xy=(0.429, 1.05), xycoords='axes fraction', xytext=(0.95, 1.05),
    arrowprops=dict(arrowstyle="<-", color='k'))

ax.annotate('', xy=(0.13, -0.2), xycoords='axes fraction', xytext=(0.7, -0.2),
    arrowprops=dict(arrowstyle="-", color='grey', linewidth=2))
ax.annotate('|', fontsize=20, xy=(0.418, 1.05), xycoords='axes fraction', va='center')
ax.annotate('|', fontsize=20, xy=(0.18, -0.2), color='grey', xycoords='axes fraction', va='center', ha='center')
ax.annotate('|', fontsize=20, xy=(0.4, -0.2), color='grey', xycoords='axes fraction', va='center', ha='center')
ax.annotate('|', fontsize=20, xy=(0.65, -0.2), color='grey', xycoords='axes fraction', va='center', ha='center')

offset = -0.3
ax.annotate('2020', fontsize=11, xy=(0.18, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax.annotate('2030', fontsize=11, xy=(0.4, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax.annotate('2040', fontsize=11, xy=(0.65, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax.annotate('year', fontsize=12, xy=(0.1, -0.2), color='grey', xycoords='axes fraction', va='center', ha='center')

ax.annotate('minimum infrastructure costs of € 57.2 Mio.', fontsize=13, ha='center', xy=(0.72, 1.095),
            xycoords='axes fraction')
ax.annotate('€ [57.2, 57.9] Mio.', fontsize=15, ha='center', xy=(0.15, 1.05),
            xycoords='axes fraction')
blue_patch = mpatches.Patch(color=c, label='Distribution of nb. of CS along CPs')
lns = l0 + l1
labs = [l.get_label() for l in lns]
ax.legend(lns + [blue_patch], labs + [blue_patch._label], loc=2, fontsize=10)
# plt.show()
plt.savefig(path + "range_30.pdf", bbox_inches="tight")
