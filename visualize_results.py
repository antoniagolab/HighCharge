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


def visualize_results(
    filename,
    dist_max,
    eta,
    charging_capacity,
    energy,
    specific_demand,
    optimization_result,
    scenario_name,
    pole_peak_cap
):
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

    # results
    # get latest result file
    list_of_files = glob.glob("results/*")
    latest_file = max(list_of_files, key=os.path.getctime)

    results = pd.read_csv(filename, header=1)
    print(results.keys())
    print(filename)

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
    plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
    plot_austrian_border = austrian_border.to_crs("EPSG:3857")
    plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y

    # turn into GeoDataframe
    results_and_geom_df_2["geometry"] = results_and_geom_df_2.centroid
    results_and_geom_df_2 = gpd.GeoDataFrame(
        results_and_geom_df_2, crs=reference_coord_sys, geometry="geometry"
    )

    # plot
    plot_results_and_geom_df_2 = results_and_geom_df_2.to_crs("EPSG:3857")

    plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
    plot_austrian_border = austrian_border.to_crs("EPSG:3857")
    plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)
    plot_results_and_geom_df_2["x"] = plot_results_and_geom_df_2.geometry.x
    plot_results_and_geom_df_2["y"] = plot_results_and_geom_df_2.geometry.y

    # scale marker sizes
    max_size = 350
    max_val = plot_results_and_geom_df["total_charging_pole_number"].max()
    fact = int(max_size/max_val)

    # classify here after 5 categories
    # lower lims + upper lims
    # low_lim = [0]

    bounds = np.linspace(0, max_val * pole_peak_cap, 6)

    print(results)
    # figure 1
    fig, ax = plt.subplots(figsize=(15, 7))
    # rc('text', usetex=True)
    plt.rcParams["font.family"] = "Franklin Gothic Book"
    plt.rcParams["font.size"] = 14
    plot_highway_geometries.plot(ax=ax, label="Austrian highway network", color='black', zorder=0)
    for ij in range(0, len(bounds)-1):
        cat = plot_results_and_geom_df[plot_results_and_geom_df["total_charging_pole_number"].isin(np.arange(bounds[ij]+pole_peak_cap, bounds[ij+1] + pole_peak_cap) / pole_peak_cap)]
        plt.scatter(
            cat["x"].to_list(),
            cat["y"].to_list(),
            s=np.array(cat["total_charging_pole_number"].to_list())
            * fact,
            color=colors[ij],
            label=str(int(bounds[ij])+ pole_peak_cap) + ' - ' + str(int(bounds[ij+1])) + ' kW',
            # edgecolors='black',
            zorder=10,
        )
        print(ij, len(cat), np.array(cat["total_charging_pole_number"].to_list())
            * fact, colors[ij])

    plot_austrian_border.plot(ax=ax, color='grey')

    ls = np.arange(
        plot_results_and_geom_df["total_charging_pole_number"].min(),
        plot_results_and_geom_df["total_charging_pole_number"].max() + 1,
    )

    # handles2, labs2 = scatter2.legend_elements(prop='sizes', num=6, alpha=0.6)
    # handles, labs = scatter.legend_elements(prop='sizes', num=6, alpha=0.6)
    #
    # labels = []
    # # n = len(labs)
    # for l in ls:
    #     labels.append("$\\mathdefault{" + str(int(l)) + "}$")

    # legend = ax.legend(
    #     handles, labs2, loc="upper left", title="Nb. of charging poles"
    # )
    # ax.add_artist(legend)
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Local peak capacities (kW)', rotation=270)
    ax.legend()
    plt.ylim([5.75e6, 6.3e6])
    ax.set_xlabel("X (EPSG:3857)")
    ax.set_ylabel("Y (EPSG:3857)")
    plt.axis('off')
    plt.text(1.1e6, 5.8e6, 'Nb. charging stations: '+ str(int(optimization_result[0]))
            + '\nNb. charging poles: ' + str(int(optimization_result[1]))+ '\nTotal infrastructure capacity (kW): ' + str(int(optimization_result[1])*pole_peak_cap), fontsize=14)
    plt.title(scenario_name, fontsize = 16)
    plt.savefig(
        "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_visualization.png"
    )
    plt.savefig(
        "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_visualization.svg"
    )
    plt.savefig(
        "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_visualization.pdf"
    )


    # plt.show()

    # TODO: write a function to create geometry for node type 2 and sum up the values of not covered energy
    # plot_results_and_geom_df_2 = plot_results_and_geom_df_2[plot_results_and_geom_df_2.not_covered > 0]
    #
    # # figure 2
    # fig, ax = plt.subplots(figsize=(15, 7))
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.size"] = 16
    # plot_highway_geometries.plot(ax=ax, label="Austrian highway network", color='black', zorder=0)
    # scatter2 = plt.scatter(
    #     plot_results_and_geom_df_2["x"].to_list(),
    #     plot_results_and_geom_df_2["y"].to_list(),
    #     s=100,
    #     c=np.array(plot_results_and_geom_df_2["not_covered"].to_list()), cmap='coolwarm',
    #     label="Charging station",
    #     edgecolors='black',
    #     zorder=10,
    # )
    #
    # plot_austrian_border.plot(ax=ax, color='grey')
    # # ax.legend(loc="lower left")
    # ax.set_xlabel("X (EPSG:3857)")
    # ax.set_ylabel("Y (EPSG:3857)")
    # plt.axis('off')
    # cb = plt.colorbar(scatter2)
    # cb.set_label('not covered energy (kW)', )
    #
    # plt.text(1.21e6, 6.2e6, 'Nb. charging stations: ' + str(int(optimization_result[0]))
    #          + '\nNb. charging poles: ' + str(int(optimization_result[1])), fontsize=16)
    # plt.title('Optimization result: ' + scenario_name, fontdict={'family': "serif"})
    # plt.savefig(
    #     "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_not_covered_demand.png"
    # )
    # plt.savefig(
    #     "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_not_covered_demand.svg"
    # )
    # plt.savefig(
    #     "results/" + latest_file.split("\\")[-1].split("_")[0] + '_' + scenario_name + "_not_covered_demand.pdf"
    # )


# if __name__ == "__main__":
#     visualize_results()
