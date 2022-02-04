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
from file_import import *

from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# colors
colors = ["#6b705c", "#2a9d8f", "#264653", "#f4a261", "#e76f51"]
# colors.reverse()

# reference coordinate system for all visualisation
reference_coord_sys = "EPSG:31287"

# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))

copy_highway_geometries = highway_geometries.drop_duplicates(subset=["highway"])

# austrian borders
austrian_border = gpd.read_file("austrian_border.shp")

# get latest result file
list_of_files = glob.glob("scenarios/*")
# latest_file = max(list_of_files, key=os.path.getctime)

charging_capacity = 350  # (kW)
#
# val = pd.read_csv(
#     "results/20220111-205736_optimization_result_charging_stations.csv"
# )  # SC

val = pd.read_csv(
    "scenarios/results v4/20220201-162601_Directed Transition_optimization_result_charging_stations.csv"
)  # SC

# energies = scenario_file.p_max_bev.to_list()
p = gpd.read_file("OGDEXT_NUTS_1_STATISTIK_AUSTRIA_NUTS2_20160101\output_BL.shp")
# p = p[p['CNTR_CODE'] == 'AT']
# p = p[p['LEVL_CODE'] == 2]


def merge_with_geom(results, energy):

    filtered_results = results[results[col_type] == "ra"]
    # osm geometries

    rest_areas = pd2gpd(
        pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
    ).sort_values(by=["on_segment", "dist_along_highway"])
    rest_areas["segment_id"] = rest_areas["on_segment"]
    rest_areas[col_type_ID] = rest_areas["nb"]
    rest_areas[col_directions] = rest_areas["evaluated_dir"]

    # merge here
    results_and_geom_df = pd.merge(
        filtered_results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
    )

    # turn into GeoDataframe
    results_and_geom_df["geometry"] = results_and_geom_df.centroid
    results_and_geom_df["total_charging_pole_number"] = np.where(
        np.array(results_and_geom_df.pYi_dir) == 0,
        np.nan,
        np.array(results_and_geom_df.pYi_dir),
    )
    results_and_geom_df = gpd.GeoDataFrame(
        results_and_geom_df, crs=reference_coord_sys, geometry="geometry"
    )
    results_and_geom_df["charging_capacity"] = (
        results_and_geom_df["total_charging_pole_number"] * energy
    )

    # plot
    plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
    # plot_results_and_geom_df = plot_results_and_geom_df[
    #     plot_results_and_geom_df.total_charging_pole_number > 0
    # ]
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
    return plot_results_and_geom_df


plot_sc_1 = merge_with_geom(val, charging_capacity)
merged = pd.merge(
    plot_sc_1, existing_infr, how="left", on=["segment_id", "name", "dir"]
)
# sub_1 = merged[merged.has_charging_station == True]
# sub_2 = merged[merged.total_charging_pole_number > 0]

merged["existing_cap"] = +merged["350kW"] * 350
merged["model_cap"] = merged["total_charging_pole_number"] * charging_capacity
merged["existing_cap"] = merged["existing_cap"].replace(np.NaN, 0)
merged["model_cap"] = merged["model_cap"].replace(np.NaN, 0)

# make classification here
merged["diff"] = merged["model_cap"] - merged["existing_cap"]

merged["difference"] = np.where(merged["diff"] < 0, 0, merged["diff"])

comp_df = merged[merged.total_charging_pole_number > 0]

max_val = comp_df["difference"].max()
bounds = [charging_capacity] + [int(round(max_val / 2, -2)), int(max_val)]
size_1 = 150  # small
size_2 = 300  # big
# plot grey , difference == 0
# plot the two classes, for where has_charging_infrastructure == True (blue)
#  plot the two classes, for where !(has_charging_infrastructure == True) (red)

plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)

bd = p.to_crs("EPSG:3857")
fig, (ax, ax1) = plt.subplots(
    2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [10, 1]}
)
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 13

# plotting NUTS 2
geoms = bd.geometry.to_list()
names = bd["NAME"].to_list()
for ij in range(0, len(bd)):
    if geoms[ij].type == "MultiPolygon":
        for g in geoms[ij]:
            ax.plot(*g.exterior.xy, color="grey", linewidth=1)
    else:
        ax.plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
    c = geoms[ij].centroid
    ax.text(c.x, c.y + 0.03e6, names[ij], color="grey")


# plotting highway network and Austrian boarder
plot_highway_geometries.plot(
    ax=ax, label="Austrian highway network", color="black", zorder=0, linewidth=1
)

# count together for the four categories all capacity
expansion_values = []


# plot_austrian_border.plot(ax=ax, color="grey", linewidth=1)

# plot the ones with no change

# plot the two classes, for where has_charging_infrastructure == True (blue);

cat = comp_df[comp_df.has_charging_station == True]
cat0 = cat[cat["difference"].isin(list(range(bounds[0], bounds[1] + 1)))]
cat1 = cat[cat["difference"].isin(list(range(bounds[1] + 1, bounds[2] + 1)))]

expansion_values.append(cat0["difference"].sum())
expansion_values.append(cat1["difference"].sum())
ax.scatter(
    cat0["x"].to_list(),
    cat0["y"].to_list(),
    s=size_1,
    color=colors[1],
    label="Expansion of existing CP by "
    + str(bounds[0])
    + " - "
    + str(bounds[1])
    + " kW",
    # edgecolors='black',
    zorder=10,
)
ax.scatter(
    cat1["x"].to_list(),
    cat1["y"].to_list(),
    s=size_2,
    color=colors[2],
    label="Expansion of existing CP by "
    + str(bounds[1] + charging_capacity)
    + " - "
    + str(bounds[2])
    + " kW",
    # edgecolors='black',
    zorder=10,
)

#  plot the two classes, for where !(has_charging_infrastructure == True) (red)

cat = comp_df[~(comp_df.has_charging_station == True)]
cat0 = cat[cat["difference"].isin(list(range(bounds[0], bounds[1] + 1)))]
cat1 = cat[cat["difference"].isin(list(range(bounds[1] + 1, bounds[2] + 1)))]
expansion_values.append(cat0["difference"].sum())
expansion_values.append(cat1["difference"].sum())
ax.scatter(
    cat0["x"].to_list(),
    cat0["y"].to_list(),
    s=size_1,
    color=colors[3],
    label="Newly installed CP with " + str(bounds[0]) + " - " + str(bounds[1]) + " kW",
    # edgecolors='black',
    zorder=10,
)
ax.scatter(
    cat1["x"].to_list(),
    cat1["y"].to_list(),
    s=size_2,
    color=colors[4],
    label="Newly installed CP with "
    + str(bounds[1] + charging_capacity)
    + " - "
    + str(bounds[2])
    + " kW",
    # edgecolors='black',
    zorder=10,
)
cat = comp_df[comp_df["difference"] == 0]
if len(cat) > 0:
    ax.scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=size_1,
        color=colors[0],
        label="No expansion of CP",
        # edgecolors='black',
        zorder=10,
    )


ax.axis("off")

ax.text(
    1.06e6,
    6.2e6,
    "Required charging infrastructure expansion\nuntil 2030 in the DT scenario",
    bbox=dict(facecolor="none", edgecolor="#d9d9d9", boxstyle="round,pad=0.5"),
    fontsize=14,
)


ax.legend(loc="lower left", bbox_to_anchor=(0.08, 1, 1, 0), ncol=2, fancybox=True)
# plt.show()


tot = sum(expansion_values)
expansion_values = [e / tot * 100 for e in expansion_values]

ax1.set_title(
    "Shares of magnitudes of capacity expansions at existing and new charging points:\n",
    loc="left",
)
ax1.broken_barh(
    [
        (0, expansion_values[0]),
        (expansion_values[0], expansion_values[1]),
        (expansion_values[0] + expansion_values[1], expansion_values[2]),
        (
            expansion_values[0] + expansion_values[1] + expansion_values[2],
            expansion_values[3],
        ),
    ],
    [10, 9],
    facecolors=colors[1:5],
    zorder=20,
)
ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax1.get_yaxis().set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_color("lightgrey")
b_box = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5")
ax1.text(
    expansion_values[0] / 2,
    22,
    str(int(round(expansion_values[0], 0))) + " %",
    bbox=b_box,
)
ax1.text(
    expansion_values[0] + expansion_values[1] / 2,
    22,
    str(int(round(expansion_values[1], 0))) + " %",
    bbox=b_box,
)
ax1.text(
    expansion_values[0] + expansion_values[1] + expansion_values[2] / 2,
    22,
    str(int(round(expansion_values[2], 0))) + " %",
    bbox=b_box,
)
ax1.text(
    expansion_values[0]
    + expansion_values[1]
    + expansion_values[2]
    + expansion_values[3] / 2,
    22,
    str(int(round(expansion_values[3], 0))) + " %",
    bbox=b_box,
)
ax1.set_ylim(5, 24)
ax1.set_xlim(0, 100)
ax1.grid(axis="x")
# plt.show()
plt.savefig("scenarios/expansion_image.pdf", bbox_inches="tight")
