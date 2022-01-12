"""
Definition of parameters for charging station allocation
"""

import numpy as np
import pandas as pd
from utils import clean_segments, filter_segments, split_by_dir, pd2gpd
from optimization import *
from variable_definitions import *
from file_import import *

# read pois_df and reindex

pois_df = pd.read_csv(
    "data/_demand_calculated.csv"
)  # file containing POIs (points of interest) which
# highway crossing and service stations (no parking spots at
# this point
pois_df["POI_ID"] = range(0, len(pois_df))
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))
links_gdf = pd2gpd(pd.read_csv("data/highway_intersections.csv"))
links_gdf = links_gdf[~(links_gdf.index == 19)]
# pois_df, segments_gdf, links_gdf = filter_segments(pois_df, segments_gdf, links_gdf,
# [ind for ind in segments_gdf.ID.to_list() if ind not in [34,44,0,1,2]])
# links_gdf.at[14, 'conn_edges'] = '[33,31]'
# links_gdf.at[21, 'conn_edges'] = '[42,46]'
# segments_gdf.at[43, 'link_0'] = np.NaN
# segments_gdf.at[45, 'link_0'] = np.NaN
#
# pois_df, segments_gdf, links_gdf = filter_segments(pois_df, segments_gdf, links_gdf,
#                                                     [0, 1, 2])
# links_gdf.at[23, 'conn_edges'] = '[46,51,49]'
# links_gdf.at[20, 'conn_edges'] = '[44,45, 43]'
# links_gdf.at[27, 'conn_edges'] = '[51,45]'
# links_gdf.at[27, 'conn_edges'] = '[51,45]'
# links_gdf.at[25, 'conn_edges'] = '[41,54]'
#
# segments_gdf.at[50, 'link_1'] = np.NaN
# segments_gdf.at[56, 'link_1'] = np.NaN
# # segments_gdf.at[46, 'link_0'] = np.NaN
# segments_gdf.at[34, 'link_0'] = np.NaN
# segments_gdf.at[43, 'link_1'] = np.NaN

# segments_gdf.at[51, 'link_0'] = np.NaN
# segments_gdf.at[31, 'link_1'] = np.NaN
#
# segments_gdf.at[37, 'link_1'] = np.NaN
# segments_gdf.at[53, 'link_0'] = np.NaN
# segments_gdf.at[47, 'link_1'] = np.NaN
# segments_gdf.at[39, 'link_1'] = np.NaN

# segments_gdf.at[1, 'link_0'] = np.NaN
# segments_gdf.at[1, 'link_1'] = np.NaN
# segments_gdf.at[0, 'link_0'] = np.NaN
pois_df, segments_gdf, links_gdf = clean_segments(links_gdf, segments_gdf, pois_df)

pois_0, pois_1 = split_by_dir(pois_df, "dir")

input_existing_infrastructure = True

pois_0[col_energy_demand] = pois_0["demand_0"]
pois_0[col_traffic_flow] = pois_0["tc_0"]
pois_1[col_energy_demand] = pois_1["demand_1"]
pois_1[col_traffic_flow] = pois_1["tc_1"]

dir_0 = pois_0
dir_1 = pois_1

dir_1 = dir_1.sort_values(by=[col_segment_id, col_distance], ascending=[True, False])
dir_1["ID"] = range(0, len(dir_1))
dir_1 = dir_1.set_index("ID")

dir = dir_0.append(dir_1)
dir = dir[
    [col_POI_ID, col_segment_id, col_directions, col_distance, col_type_ID, "pois_type"]
]
dir = dir.sort_values(by=[col_segment_id, col_POI_ID, "pois_type", col_distance])
dir = dir.drop_duplicates(subset=[col_POI_ID, col_directions])
dir["ID"] = range(0, len(dir))
dir = dir.set_index("ID")

pois_df = dir
n0 = len(dir_0)
n1 = len(dir_1)
k = len(dir)
n3 = k
# 50 kW annehmen -> + 20h durchgehende Besetzung
g = 100000  # Maximum number of charging poles at one charging station
specific_demand = 25  # (kWh/100km) average specific energy usage for 100km

acc = (
    specific_demand * 100
)  # (kWh) charged energy by a car during one charging for 100km
charging_capacity = 80.85 # (kW)
energy = charging_capacity # (kWh/h)
ec = 0.25  # (€/kWh) charging price for EV driver
e_tax = 0.15  # (€/kWh) total taxes and other charges
cx = 7000 * 3 # (€) total installation costs of charging station installation
cy = 17750  # (€) total installation costs of charging pole installation
eta = 0.9  # share of electric vehicles of car fleet
mu = 0.18  # share of cars travelling long-distance
gamma_h = 0.125   # share of cars travelling during peak hour
a = 0.69

directions_0 = dir_0[col_directions].to_list()
directions_1 = dir_1[col_directions].to_list()
# dmax = 50000
dmax = 500000

introduce_existing_infrastructure = True
no_new_infrastructure = False

# extracting all highway names to create two additional columns: "first" and "last" to indicate whether resting areas
# are first resting areas along a singular highway in "normal" direction

l1 = dir_1[col_segment_id].to_list()
l0 = dir_0[col_segment_id].to_list()
l_ext = l0
l_ext.extend(l1)
highway_names = list(set(l_ext))

maximum_dist_between_charging_stations = dmax

# read existing infrastructure


dir = pois_df
optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    cx,
    cy,
    dmax,
    eta,
    acc,
    mu,
    gamma_h,
    a,
    charging_capacity,
    specific_demand,
    introduce_existing_infrastructure,
    no_new_infrastructure,
    existing_infr_0,
    existing_infr_1,
    scenario_name="Existing infrastructure"
)