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


demand_data_0 = pd.read_csv('data/rest_area_0_input_optimization_v4.csv')
demand_data_1 = pd.read_csv('data/rest_area_1_input_optimization_v4.csv')

demand_data_0['actual_demand'] = demand_data_0[col_energy_demand] * eta * mu
demand_data_1['actual_demand'] = demand_data_1[col_energy_demand] * eta * mu


# reference coordinate system for all visualisation
reference_coord_sys = "EPSG:31287"


demand_data_0['geometry'] = demand_data_0.centroid.apply(wkt.loads)
demand_data_0 = gpd.GeoDataFrame(demand_data_0, crs=reference_coord_sys, geometry='geometry')
demand_data_1['geometry'] = demand_data_1.centroid.apply(wkt.loads)
demand_data_1 = gpd.GeoDataFrame(demand_data_1, crs=reference_coord_sys, geometry='geometry')

demand_data = demand_data_0.append(demand_data_1)
demand_data = demand_data.drop_duplicates(subset=[col_highway, col_directions, col_rest_area_name])
demand_data = demand_data.sort_values(by=[col_highway, col_distance])

actual_demand_list = []
highways = demand_data[col_highway].to_list()
dirs = demand_data[col_directions].to_list()
ra_names = demand_data[col_rest_area_name].to_list()
for ij in range(0, len(demand_data)):
    current_highway = highways[ij]
    current_dir = dirs[ij]
    current_ra_name = ra_names[ij]
    extract_0 = demand_data_0[(demand_data_0[col_highway] == current_highway) &
                              (demand_data_0[col_directions] == current_dir) &
                              (demand_data_0[col_rest_area_name] == current_ra_name)]

    extract_1 = demand_data_1[(demand_data_1[col_highway] == current_highway) &
                              (demand_data_1[col_directions] == current_dir) &
                              (demand_data_1[col_rest_area_name] == current_ra_name)]

    if len(extract_0) > 0 and len(extract_1) > 0:
        actual_demand_list.append(extract_0.actual_demand.to_list()[0] + extract_1.actual_demand.to_list()[0])
    elif len(extract_0) > 0:
        actual_demand_list.append(extract_0.actual_demand.to_list()[0])
    elif len(extract_1) > 0:
        actual_demand_list.append(extract_1.actual_demand.to_list()[0])

demand_data['actual_demand'] = actual_demand_list


# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length

copy_highway_geometries = highway_geometries.drop_duplicates(subset=["highway"])

# austrian borders
austrian_border = gpd.read_file("austrian_border.shp")

# plotting data
plot_demand_0 = demand_data_0.to_crs("EPSG:3857")
plot_demand_1 = demand_data_1.to_crs("EPSG:3857")
plot_demand = demand_data.to_crs("EPSG:3857")
plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")

# color bar
norm = colors.Normalize(
    vmin=np.min(plot_demand_0.actual_demand.to_list() + plot_demand_1.actual_demand.to_list()), vmax=np.max(plot_demand_0.actual_demand.to_list() + plot_demand_1.actual_demand.to_list())
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="Greens")

norm2 = colors.Normalize(
    vmin=plot_demand.actual_demand.min(), vmax=plot_demand.actual_demand.max()
)
cbar2 = plt.cm.ScalarMappable(norm=norm2, cmap="Greens")

# figure 1

fig, ax = plt.subplots(figsize=(15, 7))
# ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
plot_highway_geometries.plot(ax=ax)
plot_demand_0.plot(
    column="actual_demand",
    cmap="Greens",
    ax=ax,
    legend=False,
    markersize=100,
)
plot_austrian_border.plot(ax=ax)
ax_cbar = fig.colorbar(cbar, ax=ax)
ax_cbar.set_label("average demand (kWh) per day")
ax.set_xlabel("X (EPSG:3857)")
ax.set_ylabel("Y (EPSG:3857)")
ax.set_title(
    "Demand dir=0"
)
plt.savefig(
    "demand/demand_map_0.png"
)


# figure 2

fig, ax = plt.subplots(figsize=(15, 7))
plot_highway_geometries.plot(ax=ax)
plot_demand_1.plot(
    column="actual_demand",
    cmap="Greens",
    ax=ax,
    legend=False,
    markersize=100,
)
# ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
plot_austrian_border.plot(ax=ax)
ax_cbar = fig.colorbar(cbar, ax=ax)
ax_cbar.set_label("average demand (kWh) per day")
ax.set_xlabel("X (EPSG:3857)")
ax.set_ylabel("Y (EPSG:3857)")
ax.set_title(
    "Demand dir=1"
)
plt.savefig(
    "demand/demand_map_1.png"
)

# figure 3

fig, ax = plt.subplots(figsize=(15, 7))
plot_highway_geometries.plot(ax=ax)
plot_demand.plot(
    column="actual_demand",
    cmap="Greens",
    ax=ax,
    legend=False,
    markersize=100,
)
# ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
plot_austrian_border.plot(ax=ax)
ax_cbar = fig.colorbar(cbar2, ax=ax)
ax_cbar.set_label("average demand (kWh) per day")
ax.set_xlabel("X (EPSG:3857)")
ax.set_ylabel("Y (EPSG:3857)")
ax.set_title(
    "Summed up demand for both directions"
)
plt.savefig(
    "demand/demand_map.png"
)


# make histograms
plt.figure()
demand_data_0.actual_demand.hist()
plt.figure()
demand_data_1.actual_demand.hist()
plt.show()
