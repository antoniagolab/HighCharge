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
from matplotlib.ticker import MaxNLocator

# reference coordinate system for all visualisation
reference_coord_sys = "EPSG:31287"
path = "sensitivity_analyses/TC range/"
# path = "sensitivity_analyses/30perc/"
charging_capacity = 150
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

val = pd.read_csv(
    "scenarios/results v4/20220201-162601_Directed Transition_optimization_result_charging_stations.csv"
)

# energies = scenario_file.p_max_bev.to_list()
p = gpd.read_file("OGDEXT_NUTS_1_STATISTIK_AUSTRIA_NUTS2_20160101\output_BL.shp")
# p = p[p['CNTR_CODE'] == 'AT']
# p = p[p['LEVL_CODE'] == 2]


# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))

plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
# plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)


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

results = []

# range_max = 1400
# range_min = 200
# step_size = 100

range_max = 1400
range_min = 200
step_size = 100
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path + list_of_paths[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []
df["POI_ID"] = base_df["POI_ID"]
for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path + list_of_paths[ij])
    df[ranges[ij]] = temp_df.pXi
    nb_x.append(temp_df.pXi.sum())
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

merge_2 = pd.merge(df_to_plot, merged, how="left", on=["POI_ID"])

# plot

occurances = []
for ij in range(0, len(merge_2)):
    l = 0
    for r in ranges:
        if df_to_plot[r].to_list()[ij] == 1:
            l = l + 1
    occurances.append(l)


p = gpd.read_file("OGDEXT_NUTS_1_STATISTIK_AUSTRIA_NUTS2_20160101\output_BL.shp")
bd = p.to_crs("EPSG:3857")
merge_2["always_there"] = occurances
merge_3 = merge_2.copy()
merge_2 = merge_2[merge_2["always_there"] > 0]

max_occurance_val = max(occurances)
min_occurance_val = min(occurances)


colors = ['#00798c', '#d1495b', "#5abcb9"]

print("Nb. of steady PC:", len(merge_2))


fig = plt.figure(figsize=(15, 10))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 13
gs = fig.add_gridspec(2)
axs = gs.subplots(sharex=True, sharey=True)
# plotting NUTS 2
geoms = bd.geometry.to_list()
names = bd["NAME"].to_list()
for ij in range(0, len(bd)):
    if geoms[ij].type == "MultiPolygon":
        for g in geoms[ij]:
            axs[0].plot(*g.exterior.xy, color="grey", linewidth=1)
            axs[1].plot(*g.exterior.xy, color="grey", linewidth=1)
    else:
        axs[0].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
        axs[1].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
    c = geoms[ij].centroid
    axs[0].text(c.x, c.y + 0.03e6, names[ij], color="grey")
    axs[1].text(c.x, c.y + 0.03e6, names[ij], color="grey")


# plotting highway network and Austrian boarder
plot_highway_geometries.plot(
    ax=axs[0], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_highway_geometries.plot(
    ax=axs[1],  label="Austrian highway network", color="black", zorder=0, linewidth=1
)
axs[0].axis("off")
axs[1].axis("off")

# cat 0: "1-4"
plot_df = merge_2[merge_2["always_there"].isin(range(1, 5))]
axs[0].scatter(
    plot_df["x"], plot_df["y"], s=90, color="#ced4da", label="1-4 occurances", zorder=20
)

# cat 1: "5-9"
plot_df = merge_2[merge_2["always_there"].isin(range(5, 10))]
axs[0].scatter(
    plot_df["x"], plot_df["y"], s=90, color="#adb5bd", label="5-9 occurances", zorder=20
)

# cat 2: "10-12"
plot_df = merge_2[merge_2["always_there"].isin(range(10, 13))]
axs[0].scatter(
    plot_df["x"], plot_df["y"], s=90, color="#6c757d", label="10-12 occurances", zorder=20
)

# cat 3: "13"
plot_df = merge_2[merge_2["always_there"]==13]
print(len(plot_df), " CP present in all")
axs[0].scatter(
    plot_df["x"], plot_df["y"], s=90, color="#212529", label="occurance in every SA model run", zorder=20
)

axs[0].text(
    1.06e6,
    6.2e6,
    "Constantly present CP locations during range SA",
    bbox=dict(facecolor="none", edgecolor="#d9d9d9", boxstyle="round,pad=0.5"),
    fontsize=11,
)

axs[0].legend(loc="lower left", bbox_to_anchor=(0.15, 0.97, 1, 0), ncol=2, fancybox=True)

# PLOT axis 1
# A: has charging station = YES
# B: always_there = max_occurances_value

# cat 0: A + B

cat = merge_3[(merge_3['has_charging_station'] == True) & (merge_3['always_there'] == 13)]
print(len(cat), " CP also present in existing")

axs[1].scatter(
    cat["x"], cat["y"], s=90, color=colors[0], label="part of existing infr. and constant in SA", zorder=15
)

# cat 1: A
cat = merge_3[(merge_3['has_charging_station'] == True) & (merge_3['always_there'] != 13)]

axs[1].scatter(
    cat["x"], cat["y"], s=90, color=colors[2], label="part of existing infr. ", zorder=20
)

# cat: B
cat = merge_3[(merge_3['has_charging_station'] != True) & (merge_3['always_there'] == 13)]

axs[1].scatter(
    cat["x"], cat["y"], s=90, color=colors[1], label="constant occurance during SA", zorder=20
)

axs[1].legend(loc="lower left", bbox_to_anchor=(0.05, 0.95, 1, 0), ncol=2, fancybox=True)
plt.subplots_adjust(
                    hspace=0.1)

axs[1].text(
    1.06e6,
    6.2e6,
    "Comparison to existing infrastructure",
    bbox=dict(facecolor="none", edgecolor="#d9d9d9", boxstyle="round,pad=0.5"),
    fontsize=11,
)
# plt.show()
plt.savefig("sensitivity_analyses/steady_CS.pdf", bbox_inches="tight")

