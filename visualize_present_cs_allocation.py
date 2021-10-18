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

# reference coordinate system
reference_coord_sys = "EPSG:31287"

# existing charging stations
rest_areas_0 = pd.read_csv('data/rest_areas_0_charging_stations.csv')
rest_areas_1 = pd.read_csv('data/rest_areas_1_charging_stations.csv')

rest_areas_0['geometry'] = rest_areas_0.centroid.apply(wkt.loads)
rest_areas_0 = gpd.GeoDataFrame(rest_areas_0, crs=reference_coord_sys, geometry='geometry')

rest_areas_1['geometry'] = rest_areas_1.centroid.apply(wkt.loads)
rest_areas_1 = gpd.GeoDataFrame(rest_areas_1, crs=reference_coord_sys, geometry='geometry')


# highway geometries
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v6.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries['length'] = highway_geometries.geometry.length

copy_highway_geometries = highway_geometries.drop_duplicates(subset=['highway'])


# austrian borders
austrian_border = gpd.read_file('austrian_border.shp')

# how much energy demand covered
rest_areas = rest_areas_0.append(rest_areas_1)
rest_areas = rest_areas.sort_values(by=['highway', 'has_charging_station'])
rest_areas = rest_areas.drop_duplicates(subset=[col_highway, col_rest_area_name, col_directions])
rest_areas['has_charging_station'] = rest_areas['has_charging_station'].fillna(False)
rest_areas['capacity_per_day'] = np.where(rest_areas['has_charging_station'],
                                          ((acc * hours_of_constant_charging/(acc/charging_capacity)) * rest_areas['cs_below_50kwh'] + (acc * hours_of_constant_charging/(acc/150)) * rest_areas['cs_above_50kwh']),
                                                 np.NaN)

print(np.sum(np.where(rest_areas['has_charging_station'], rest_areas['cs_below_50kwh'] + rest_areas['cs_above_50kwh'], 0)))

highway_capacities = rest_areas.groupby(col_highway).sum()
highway_capacities['ind'] = range(0, len(highway_capacities))
highway_capacities[col_highway] = highway_capacities.index
highway_capacities = highway_capacities.set_index('ind')

copy_highway_geometries = pd.merge(copy_highway_geometries[[col_highway, 'length', 'geometry']], highway_capacities[['highway', 'capacity_per_day']],
                      left_on=col_highway, right_on=col_highway)
copy_highway_geometries['capacity_per_km_per_day'] = copy_highway_geometries['capacity_per_day']/copy_highway_geometries['length']
copy_highway_geometries = copy_highway_geometries.fillna(0.0)

# prepare dataframes to plot
plot_rest_areas = rest_areas.to_crs(crs="EPSG:3857")
plot_rest_areas = plot_rest_areas[plot_rest_areas['has_charging_station']]
plot_highway_geometries = highway_geometries.to_crs(crs="EPSG:3857")
plot_copy_highway_geometries = copy_highway_geometries.to_crs(crs="EPSG:3857")
plot_austrian_borders = austrian_border.to_crs(crs="EPSG:3857")

norm1 = colors.Normalize(vmin=0, vmax=plot_rest_areas['cs_below_50kwh'].max())
norm2 = colors.Normalize(vmin=0, vmax=0.5)
cbar = plt.cm.ScalarMappable(norm=norm1, cmap='GnBu')
cbar2 = plt.cm.ScalarMappable(norm=norm2, cmap='terrain')

# figure 1
fig, ax = plt.subplots(figsize=(15, 7))
plot_rest_areas.plot(column='cs_below_50kwh', cmap='GnBu', ax=ax, legend=False, markersize=100, vmin=0)
plot_highway_geometries.plot(ax=ax)
plot_highway_geometries['null'] = [0]*len(plot_highway_geometries)
plot_austrian_borders.plot(ax=ax)
ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
ax_cbar = fig.colorbar(cbar, ax=ax)
ax_cbar.set_label('charging poles')
ax.set_xlabel('X (EPSG:3857)')
ax.set_ylabel('Y (EPSG:3857)')
ax.set_title('Status Quo: 50kWh charging stations')
plt.savefig('existing_infrastructure/installed_charging_stations.png')
plt.show()

# figure 2 - charging capacity per km illustrated in line thickness
fig, ax2 = plt.subplots(figsize=(15, 7))
plot_highway_geometries.plot(column='null', ax=ax2, cmap='terrain', linewidth=5, legend=False)
plot_copy_highway_geometries.plot(column='capacity_per_km_per_day', cmap='terrain', ax=ax2, legend=False, linewidth=5, vmax=0.5)
plot_austrian_borders.plot(ax=ax2)
ctx.add_basemap(ax2, url=ctx.providers.Stamen.TonerLite)
ax_cbar2 = fig.colorbar(cbar2, ax=ax2)
ax_cbar2.set_label('charging capacity per day per km [kWh/(km24h)]')
ax2.set_xlabel('X (EPSG:3857)')
ax2.set_ylabel('Y (EPSG:3857)')
ax2.set_title('Charging capacity for 24h')
plt.savefig('existing_infrastructure/installed_capacity_per_km_per_day.png')
plt.show()

print("Total available capacity: " + str(plot_copy_highway_geometries.capacity_per_day.sum()) + " kW")
print(plot_rest_areas.cs_below_50kwh.sum())