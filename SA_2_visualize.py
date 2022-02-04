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

path = "sensitivity_analyses/epsilon_increase/"

list_of_paths = [
    "20220202-125929_ev share - epsilon SC10_3_optimization_result_charging_stations.csv",
    "20220202-131134_ev share - epsilon SC20_3_optimization_result_charging_stations.csv",
    "20220202-132238_ev share - epsilon SC30_3_optimization_result_charging_stations.csv",
    "20220202-133352_ev share - epsilon SC40_3_optimization_result_charging_stations.csv",
    "20220202-134503_ev share - epsilon SC50_3_optimization_result_charging_stations.csv",
    "20220202-135620_ev share - epsilon SC60_3_optimization_result_charging_stations.csv",
    "20220202-140806_ev share - epsilon SC70_3_optimization_result_charging_stations.csv",
    "20220202-173358_ev share - epsilon SC80_3_optimization_result_charging_stations.csv",
    "20220202-162803_ev share - epsilon SC90_3_optimization_result_charging_stations.csv",
    "20220203-084114_ev share - epsilon SC100_3_optimization_result_charging_stations.csv"

]

results = []

range_max = 100
range_min = 10
step_size = 10
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path + list_of_paths[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []
nb_y = []
for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path + list_of_paths[ij])

    nb_x.append(temp_df.pXi.sum())
    nb_y.append(temp_df.pYi_dir.sum())
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

# plot

fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True
# ax2 = ax1.twinx()
plt.xlim([range_min - step_size, range_max + step_size])
ax1.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#457b9d"
#
# df_to_plot.boxplot(
#     ax=ax,
#     widths=(50),
#     notch=True,
#     patch_artist=True,
#     boxprops=dict(facecolor=c, color=c),
#     capprops={"color": c, "linewidth": 2},
#     whiskerprops={"color": c, "linewidth": 2},
#     flierprops={"color": c, "markeredgewidth": 2, "markeredgecolor": c},
#     medianprops=dict(color=c),
#     positions=ranges,
#     labels=ranges,
# )


ax1.set_xlabel("BEV share (%)", fontname="Franklin Gothic Book", fontsize=12)
ax1.grid()
# ax1.grid(True)
# ax1.set_ylabel("# charging stations", color="#1d3557", fontsize=10)
ax1.set_ylabel("Nb. charging points", labelpad=12, fontname="Franklin Gothic Book", fontsize=12)
ax1.set_ylim([min(nb_x)-2, max(nb_x) + 2])

# ax1.grid(False)
# l1 = ax1.plot(
#     list(ranges),
#     nb_y,
#     marker='o',
#     color="#457b9d",
#     linewidth=2,
#     label="# charging points",
# )

l0 = ax1.plot(
    list(ranges),
    nb_x,
    marker="o",
    color="#723d46",
    linewidth=2,
    label="# charging stations",
)


# ax2.set_ylim([36, 54])
# ax1.set_ylim([0, 2500])

# ax1.tick_params(axis="y", colors="#1d3557", labelsize=10)
# ax2.tick_params(axis="y", colors="#723d46", labelsize=10)

# ax2.spines["left"].set_color("#1d3557")
# ax1.spines["left"].set_color("#1d3557")  # setting up Y-axis tick color to red
# ax2.spines["right"].set_color("#723d46")
# ax2.spines["top"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.set_yticks(np.linspace(34, 54, 11))
# ax1.set_yticks(np.linspace(0, 2500, 11))

lns = l0
labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0, fontsize=10)
# ax2.set_xticks(ranges)
ax1.set_xticks(ranges)
# plt.show()


sens_data = pd.read_csv("sensitivity_analyses/sensitivity_anal_epsilon_part_1.csv")

perc_non_covered = np.array(sens_data.perc_not_charged) * 100
installed_caps = np.array(sens_data.nb_poles) * (350 / 1000)

# plot 2
# fig, ax = plt.subplots(figsize=(8, 3.5))

# ax2 = ax.twinx()
# ax2.set_ylim([0,30])
ax3.bar(list(ranges), installed_caps, width=8, zorder=20, color="#f4a261")
# ax2.plot([0] + list(ranges) + [110], list(perc_non_covered) + [perc_non_covered[0]]*2 , color='#233d4d', linestyle='--', linewidth=2)
ax3.set_xlim([range_min - step_size, range_max + step_size])
ax3.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#457b9d"

ax3.set_xlabel("BEV share (%)", fontsize=12, fontname="Franklin Gothic Book")
ax3.grid("off")
ax3.grid(True)
ax3.set_ylabel("installed capacity (MW)", fontname="Franklin Gothic Book", fontsize=12)
# ax2.set_ylabel(
#    "not covered energy demand (%)", color="#723d46", rotation=-90, labelpad=12, fontsize=12
# )
ax3.text(50,800, '€ [22, 201] Mio.', fontsize=14, ha='right',bbox=dict(boxstyle="round",
                   ec='w',
                   fc='w',
                   ),
            )
ax3.tick_params(axis="y", labelsize=10)
# ax2.tick_params(axis="y", colors="#723d46", labelsize=10)

# ax2.spines["left"].set_color("#1d3557")
# ax2.spines["right"].set_color("#723d46")
# ax2.spines["top"].set_visible(False)
# ax.spines["top"].set_visible(False)
# x.spines[].set_visible(False)
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.set_xticks(ranges)
ax3.set_xticks(ranges)
# ax2.grid(False)
# ax.grid(False)
ax1.annotate('', xy=(0, -0.25), xycoords='axes fraction', xytext=(1, -0.25),
    arrowprops=dict(arrowstyle="-", color='grey', linewidth=2))
ax3.annotate('', xy=(0, -0.25), xycoords='axes fraction', xytext=(1, -0.25),
    arrowprops=dict(arrowstyle="-", color='grey', linewidth=2))
ax1.annotate('|', fontsize=20, xy=(0.02, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')
ax1.annotate('|', fontsize=20, xy=(0.3, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')
ax1.annotate('|', fontsize=20, xy=(0.95, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')

ax3.annotate('|', fontsize=20, xy=(0.02, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')
ax3.annotate('|', fontsize=20, xy=(0.3, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')
ax3.annotate('|', fontsize=20, xy=(0.92, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')

offset = -0.35
ax1.annotate('2020', fontsize=11, xy=(0.02, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax1.annotate('2030', fontsize=11, xy=(0.3, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax1.annotate('2040', fontsize=11, xy=(0.95, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax3.annotate('2020', fontsize=11, xy=(0.02, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax3.annotate('2030', fontsize=11, xy=(0.3, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax3.annotate('2040', fontsize=11, xy=(0.92, offset), color='grey', xycoords='axes fraction', va='center', ha='center')
ax1.annotate('year', fontsize=12, xy=(-0.1, -0.25), color='grey', xycoords='axes fraction', va='center', ha='center')

ax1.set_yticklabels(labels=list(range(42, 60, 2)), fontname="Franklin Gothic Book")
ax3.set_yticklabels(labels=list(range(0, 900, 100)), fontname="Franklin Gothic Book")
ax1.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")
ax3.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")

plt.tight_layout()
plt.savefig(path + "infr_change.pdf", bbox_inches="tight")
# plt.show()
