import pandas as pd
import geopandas as gpd
import glob
import os
from shapely import wkt
from optimization_parameters import *
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def visualize_results(
    filename,
    maximum_dist_between_charging_stations=maximum_dist_between_charging_stations,
    eta=eta,
    ec=ec,
    acc=acc,
    charging_capacity=charging_capacity,
    cars_per_day=cars_per_day,
    energy=energy,
    specific_demand=specific_demand,
):
    # reference coordinate system for all visualisation
    reference_coord_sys = "EPSG:31287"

    # highway geometries
    highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
    highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
    highway_geometries = gpd.GeoDataFrame(highway_geometries)
    highway_geometries = highway_geometries.set_crs(reference_coord_sys)
    highway_geometries["length"] = highway_geometries.geometry.length

    copy_highway_geometries = highway_geometries.drop_duplicates(subset=["highway"])

    # austrian borders
    austrian_border = gpd.read_file("austrian_border.shp")

    # results
    # get latest result file
    list_of_files = glob.glob("results/*")
    latest_file = max(list_of_files, key=os.path.getctime)

    results = pd.read_csv(filename, header=1)

    # osm geometries
    input_0 = pd.read_csv("data/rest_area_0_input_optimization.csv")
    input_1 = pd.read_csv("data/rest_area_1_input_optimization.csv")
    input_data = input_0.append(input_1)
    input_data = input_data.drop_duplicates(subset=[col_rest_area_name, col_directions])
    input_data = input_data[
        [col_highway, col_rest_area_name, col_directions, "centroid"]
    ]

    # merge here
    results_and_geom_df = pd.merge(
        results, input_data, on=[col_highway, col_rest_area_name, col_directions]
    )

    # turn into GeoDataframe
    results_and_geom_df["geometry"] = results_and_geom_df.centroid.apply(wkt.loads)
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

    highway_capacities = results_and_geom_df.groupby(col_highway).sum()
    highway_capacities["ind"] = range(0, len(highway_capacities))
    highway_capacities[col_highway] = highway_capacities.index
    highway_capacities = highway_capacities.set_index("ind")
    print(highway_capacities.keys())
    copy_highway_geometries = pd.merge(
        copy_highway_geometries[[col_highway, "length", "geometry"]],
        highway_capacities[["highway", "charging_capacity"]],
        left_on=col_highway,
        right_on=col_highway,
    )
    copy_highway_geometries["capacity_per_km_per_day"] = (
        copy_highway_geometries["charging_capacity"] / copy_highway_geometries["length"]
    )
    copy_highway_geometries = copy_highway_geometries.fillna(0.0)

    # plot
    plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
    plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
    plot_austrian_border = austrian_border.to_crs("EPSG:3857")
    plot_copy_highway_geometries = copy_highway_geometries.to_crs(crs="EPSG:3857")
    plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)

    # color bar
    norm1 = colors.Normalize(
        vmin=0, vmax=plot_results_and_geom_df.total_charging_pole_number.max()
    )
    cbar1 = plt.cm.ScalarMappable(norm=norm1, cmap="Reds")
    norm2 = colors.Normalize(
        vmin=0, vmax=plot_results_and_geom_df.charging_capacity.max()
    )
    cbar2 = plt.cm.ScalarMappable(norm=norm2, cmap="GnBu")
    norm3 = colors.Normalize(vmin=0, vmax=0.5)
    cbar3 = plt.cm.ScalarMappable(norm=norm3, cmap="terrain")
    print(plot_copy_highway_geometries.capacity_per_km_per_day.max())
    # figure 1
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_highway_geometries.plot(ax=ax)
    plot_results_and_geom_df.plot(
        column="total_charging_pole_number",
        cmap="Reds",
        ax=ax,
        legend=False,
        markersize=100,
        vmin=0,
    )
    plot_austrian_border.plot(ax=ax)
    # ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
    ax_cbar = fig.colorbar(cbar1, ax=ax)
    ax_cbar.set_label("number of charging poles")
    ax.set_xlabel("X (EPSG:3857)")
    ax.set_ylabel("Y (EPSG:3857)")
    ax.set_title(
        "Optimized charging pole allocation at highway resting areas in Austria; dmax="
        + str(maximum_dist_between_charging_stations)
        + " km; epsilon="
        + str(eta)
        + "; \n cel="
        + str(ec)
        + "; "
        + str(charging_capacity)
        + "; specific_demand = "
        + str(specific_demand)
    )
    plt.savefig(
        "results/" + latest_file.split("\\")[-1].split("_")[0] + "_visualization.png"
    )

    # figure 2
    fig, ax2 = plt.subplots(figsize=(15, 7))
    plot_highway_geometries.plot(ax=ax2)
    plot_results_and_geom_df.plot(
        column="charging_capacity",
        cmap="GnBu",
        ax=ax2,
        legend=False,
        markersize=100,
        vmin=0,
    )
    plot_austrian_border.plot(ax=ax2)
    # ctx.add_basemap(ax2, url=ctx.providers.Stamen.TonerLite)
    ax_cbar = fig.colorbar(cbar2, ax=ax2)
    ax_cbar.set_label("charging capacity per day (kWh/24h)")
    ax2.set_xlabel("X (EPSG:3857)")
    ax2.set_ylabel("Y (EPSG:3857)")
    ax2.set_title("Charging capacity per day")
    plt.savefig(
        "results/"
        + latest_file.split("\\")[-1].split("_")[0]
        + "_charging_capacity_per_day.png"
    )

    # figure 2 - charging capacity per km illustrated in line thickness
    fig, ax3 = plt.subplots(figsize=(15, 7))
    plot_highway_geometries.plot(
        column="null", ax=ax3, cmap="terrain", linewidth=5, legend=False
    )
    plot_copy_highway_geometries.plot(
        column="capacity_per_km_per_day",
        cmap="terrain",
        ax=ax3,
        legend=False,
        linewidth=5,
        vmax=0.5,
    )
    plot_austrian_border.plot(ax=ax3)
    ctx.add_basemap(ax3, url=ctx.providers.Stamen.TonerLite)
    ax_cbar2 = fig.colorbar(cbar3, ax=ax3)
    ax_cbar2.set_label("charging capacity per day per km [kWh/(km24h)]")
    ax3.set_xlabel("X (EPSG:3857)")
    ax3.set_ylabel("Y (EPSG:3857)")
    ax3.set_title("Charging capacity for 24h")
    plt.savefig(
        "results/"
        + latest_file.split("\\")[-1].split("_")[0]
        + "_charging_capacity_per_km.png"
    )

    print(
        "Total available capacity: "
        + str(plot_copy_highway_geometries.charging_capacity.sum())
        + " kW"
    )


if __name__ == "__main__":
    visualize_results()
