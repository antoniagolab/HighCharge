import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
from optimization_parameters import *

# assumed parameters
specific_energy_demand_per_car = 0.2    # (kWh/km)

# reference coordinate system for all calculations
reference_coord_sys = "EPSG:31287"

# get rest areas
rest_areas_0 = pd.read_csv('data/rest_areas_0_with_traffic_flow_v2.csv')
rest_areas_1 = pd.read_csv('data/rest_areas_1_with_traffic_flow_v2.csv')

rest_areas_0 = rest_areas_0[~(rest_areas_0.asfinag_type == 2)]
rest_areas_1 = rest_areas_1[~(rest_areas_1.asfinag_type == 2)]


# highways
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v6.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_names = list(set(rest_areas_0.highway.to_list() + rest_areas_1.highway.to_list()))

for name in highway_names:
    # normal direction
    extract_rest_areas_0 = rest_areas_0[rest_areas_0.highway == name]
    extract_rest_areas_0 = extract_rest_areas_0.sort_values(by="dist_along_highway")
    dists = extract_rest_areas_0.dist_along_highway.to_list()
    traffic_flows = extract_rest_areas_0.traffic_flow.to_list()
    inds = extract_rest_areas_0.index.to_list()
    dist_before = 0
    for ij in range(0, len(extract_rest_areas_0)):
        dist = dists[ij] - dist_before
        dist_before = dists[ij]
        rest_areas_0.at[inds[ij], 'energy_demand'] = (dist/1000) * specific_energy_demand_per_car * traffic_flows[ij]

    # inverse direction
    extract_rest_areas_1 = rest_areas_1[rest_areas_1.highway == name]
    extract_rest_areas_1 = extract_rest_areas_1.sort_values(by="dist_along_highway")
    dists = extract_rest_areas_1.dist_along_highway.to_list()
    traffic_flows = extract_rest_areas_1.traffic_flow.to_list()
    inds = extract_rest_areas_1.index.to_list()
    dist_before = highway_geometries[highway_geometries.highway == name].geometry.to_list()[0].length
    for ij in np.arange(len(extract_rest_areas_1)-1, -1, -1):
        dist = dist_before - dists[ij]
        dist_before = dists[ij]
        rest_areas_1.at[inds[ij], 'energy_demand'] = (dist/1000) * specific_energy_demand_per_car * traffic_flows[ij]


rest_areas_0.to_csv("data/rest_area_0_input_optimization_v4.csv", index=False)
rest_areas_1.to_csv("data/rest_area_1_input_optimization_v4.csv", index=False)
