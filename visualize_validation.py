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
colors = ["#5f0f40", "#9a031e", "#E9C46A", "#e36414", "#0f4c5c"]
colors.reverse()

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

charging_capacity = 150  # (kW)

val = pd.read_csv(
    "validation_results/20220120-134318_optimization_result_charging_stations.csv"
)  # SC

# energies = scenario_file.p_max_bev.to_list()


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
sub_1 = merged[merged.has_charging_station == True]
sub_2 = merged[merged.total_charging_pole_number > 0]

sub_1["installed_cap"] = (
    sub_1["50kW"] * 50
    + sub_1["75kW"] * 75
    + sub_1["150kW"] * 150
    + sub_1["350kW"] * 350
)
sub_2["installed_cap"] = sub_2["total_charging_pole_number"] * charging_capacity


plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)
min_size = 30
max_size = 150
max_val = max([sub_1["installed_cap"].max(), sub_1["installed_cap"].max()])
fact = max_size / max_val
sizes = list(np.linspace(55, 150, 5))

bounds = np.linspace(0, max_val, 6)
cm = 1 / 2.54
bounds[0] = 50
fig = plt.figure(figsize=(15, 10))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
plot_highway_geometries.plot(
    ax=axs[0], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_highway_geometries.plot(
    ax=axs[1], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_austrian_border.plot(ax=axs[0], color="grey", linewidth=1)
plot_austrian_border.plot(ax=axs[1], color="grey", linewidth=1)

for ij in range(0, len(bounds) - 1):
    cat = sub_1[sub_1["installed_cap"].isin(np.arange(bounds[ij], bounds[ij + 1]))]
    axs[0].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=sizes[ij],
        color=colors[ij],
        label=str(int(bounds[ij])) + " - " + str(int(bounds[ij + 1])) + " kW",
        # edgecolors='black',
        zorder=10,
    )

for ij in range(0, len(bounds) - 1):
    cat = sub_2[sub_2["installed_cap"].isin(np.arange(bounds[ij], bounds[ij + 1]))]
    axs[1].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=sizes[ij],
        color=colors[ij],
        label=str(int(bounds[ij])) + " - " + str(int(bounds[ij + 1])) + " kW",
        # edgecolors='black',
        zorder=10,
    )
axs[0].axis("off")
axs[1].axis("off")
axs[0].text(
    1.07e6,
    6.2e6,
    "Existing infrastructure",
    bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=0.4'),
    fontsize=16,
)
axs[1].text(
    1.07e6,
    6.2e6,
    "Model output",
    bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=0.4'),
    fontsize=16,
)

axs[0].legend(loc="lower left", bbox_to_anchor=(0.06, 1, 1, 0), ncol=3, fancybox=True)
# plt.show()
plt.savefig('validation_results/comparison_image.pdf', bbox_inches="tight")