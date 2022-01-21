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

# colors
colors = ['#5f0f40', '#9a031e', '#E9C46A', '#e36414', '#0f4c5c']
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

charging_capacity = 350 # (kW)


scenario_file = pd.read_csv('scenarios/scenario_parameters.csv')

scenario_names = scenario_file['scenario name'].to_list()


sc_1 = pd.read_csv('scenarios/20220114-133355_optimization_result_charging_stations.csv')  # SC
sc_2 = pd.read_csv('scenarios/20220114-140333_optimization_result_charging_stations.csv')  # TF
sc_3 = pd.read_csv('scenarios/20220114-143350_optimization_result_charging_stations.csv')  # DT
sc_4 = pd.read_csv('scenarios/20220114-145749_optimization_result_charging_stations.csv')  # GD

energies = scenario_file.p_max_bev.to_list()


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
    results_and_geom_df_2 = pd.merge(
        results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
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
    plot_results_and_geom_df = plot_results_and_geom_df[
        plot_results_and_geom_df.total_charging_pole_number > 0
    ]
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
    return plot_results_and_geom_df


plot_sc_1 = merge_with_geom(sc_1, energies[0])
plot_sc_2 = merge_with_geom(sc_2, energies[1])
plot_sc_3 = merge_with_geom(sc_3, energies[2])
plot_sc_4 = merge_with_geom(sc_4, energies[3])
plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)
max_size = 60
max_val = max([plot_sc_1["total_charging_pole_number"].max(), plot_sc_2["total_charging_pole_number"].max(),
               plot_sc_3["total_charging_pole_number"].max(), plot_sc_4["total_charging_pole_number"].max()])
fact = int(max_size / max_val)

bounds = np.linspace(0, max_val * charging_capacity, 6)
cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 20*cm))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 8
plot_highway_geometries.plot(ax=axs[0], label="Austrian highway network", color='black', zorder=0, linewidth=1)
plot_highway_geometries.plot(ax=axs[1], label="Austrian highway network", color='black', zorder=0, linewidth=1)
plot_highway_geometries.plot(ax=axs[2], label="Austrian highway network", color='black', zorder=0, linewidth=1)
plot_highway_geometries.plot(ax=axs[3], label="Austrian highway network", color='black', zorder=0, linewidth=1)

plot_austrian_border.plot(ax=axs[0], color='grey', linewidth=1)
plot_austrian_border.plot(ax=axs[1], color='grey', linewidth=1)
plot_austrian_border.plot(ax=axs[2], color='grey', linewidth=1)
plot_austrian_border.plot(ax=axs[3], color='grey', linewidth=1)


for ij in range(0, len(bounds) - 1):
    cat = plot_sc_1[plot_sc_1["total_charging_pole_number"].isin(
        np.arange(bounds[ij] + charging_capacity, bounds[ij + 1] + charging_capacity) / charging_capacity)]
    axs[0].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=np.array(cat["total_charging_pole_number"].to_list())
          * fact,
        color=colors[ij],
        label=str(int(bounds[ij]) + charging_capacity) + ' - ' + str(int(bounds[ij + 1])) + ' kW',
        # edgecolors='black',
        zorder=10,
    )

for ij in range(0, len(bounds) - 1):
    cat = plot_sc_2[plot_sc_2["total_charging_pole_number"].isin(
        np.arange(bounds[ij] + charging_capacity, bounds[ij + 1] + charging_capacity) / charging_capacity)]
    axs[1].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=np.array(cat["total_charging_pole_number"].to_list())
          * fact,
        color=colors[ij],
        label=str(int(bounds[ij]) + charging_capacity) + ' - ' + str(int(bounds[ij + 1])) + ' kW',
        # edgecolors='black',
        zorder=10,
    )

for ij in range(0, len(bounds) - 1):
    cat = plot_sc_3[plot_sc_3["total_charging_pole_number"].isin(
        np.arange(bounds[ij] + charging_capacity, bounds[ij + 1] + charging_capacity) / charging_capacity)]
    axs[2].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=np.array(cat["total_charging_pole_number"].to_list())
          * fact,
        color=colors[ij],
        label=str(int(bounds[ij]) + charging_capacity) + ' - ' + str(int(bounds[ij + 1])) + ' kW',
        # edgecolors='black',
        zorder=10,
    )

for ij in range(0, len(bounds) - 1):
    cat = plot_sc_4[plot_sc_4["total_charging_pole_number"].isin(
        np.arange(bounds[ij] + charging_capacity, bounds[ij + 1] + charging_capacity) / charging_capacity)]
    axs[3].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=np.array(cat["total_charging_pole_number"].to_list())
          * fact,
        color=colors[ij],
        label=str(int(bounds[ij]) + charging_capacity) + ' - ' + str(int(bounds[ij + 1])) + ' kW',
        # edgecolors='black',
        zorder=10,
    )

axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[3].axis('off')
axs[0].legend(loc="lower left", bbox_to_anchor=(-0.3, 1, 1, 0),
          ncol=3, fancybox=True)
# plt.show()
plt.savefig('probe.pdf')



