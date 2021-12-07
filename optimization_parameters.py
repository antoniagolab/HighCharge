"""

Definition of parameters for charging station allocation

"""

import numpy as np
import pandas as pd
from utils import clean_segments, filter_segments, split_by_dir, pd2gpd
# read pois_df and reindex

pois_df = pd.read_csv('data/_demand_calculated.csv')    # file containing POIs (points of interest) which
                                                        # highway crossing and service stations (no parking spots at
                                                        # this point
pois_df['POI_ID'] = range(0, len(pois_df))
segments_gdf = pd2gpd(pd.read_csv('data/highway_segments.csv'))
links_gdf = pd2gpd(pd.read_csv('data/highway_intersections.csv'))
links_gdf = links_gdf[~(links_gdf.index==19)]
# pois_df, segments_gdf, links_gdf = filter_segments(pois_df, segments_gdf, links_gdf,
#                                                     [ind for ind in segments_gdf.ID.to_list() if ind not in [34,44,0,1,2]])
# links_gdf.at[14, 'conn_edges'] = '[33,31]'
# links_gdf.at[21, 'conn_edges'] = '[42,46]'
# segments_gdf.at[43, 'link_0'] = np.NaN
# segments_gdf.at[45, 'link_0'] = np.NaN


# TODO: das probieren:
# pois_df, segments_gdf, links_gdf = filter_segments(pois_df, segments_gdf, links_gdf,
#                                                     [34, 43, 44, 45])

# # links_gdf.at[17, 'conn_edges'] = '[38,36]'
#
# segments_gdf.at[34, 'link_0'] = np.NaN
# segments_gdf.at[44, 'link_1'] = np.NaN
# segments_gdf.at[45, 'link_1'] = np.NaN
# segments_gdf.at[43, 'link_1'] = np.NaN


# segments_gdf.at[31, 'link_1'] = np.NaN

#
#segments_gdf.at[37, 'link_1'] = np.NaN
# segments_gdf.at[53, 'link_0'] = np.NaN
# segments_gdf.at[47, 'link_1'] = np.NaN
# segments_gdf.at[39, 'link_1'] = np.NaN

# segments_gdf.at[1, 'link_0'] = np.NaN
# segments_gdf.at[1, 'link_1'] = np.NaN
# segments_gdf.at[0, 'link_0'] = np.NaN
pois_df, segments_gdf, links_gdf = clean_segments(links_gdf, segments_gdf, pois_df)

pois_0, pois_1 = split_by_dir(pois_df, 'dir')

# TODO np.Nan values need to be filled!!
# dir_0 = pd.read_csv(
#     "data/rest_area_0_input_optimization_v4.csv"
# )  # file containing service stations for both directions + for "normal" direction
# dir_1 = pd.read_csv(
#     "data/rest_area_1_input_optimization_v4.csv"
# )  # file containing service stations for both directions + for "inverse" direction

col_energy_demand = (
    "energy_demand"  # specification of column name for energy demand per day
)
col_directions = "dir"  # specification of column name for directions: 0 = 'normal'; 1 = 'inverse'; 2 = both directions
col_rest_area_name = "name"  # specification of column nameholding names of rest areas
col_traffic_flow = "traffic_flow"
col_type = "asfinag_type"
# col_position = "asfinag_position"
col_highway = "highway"
col_has_cs = "has_charging_station"
col_distance = "dist_along_segment"
col_segment_id = 'segment_id'
input_existing_infrastructure = False
col_type_ID = 'type_ID'
col_POI_ID = 'POI_ID'
pois_0[col_energy_demand] = pois_0['demand_0']
pois_1[col_energy_demand] = pois_1['demand_1']
dir_0 = pois_0
dir_1 = pois_1

dir_1 = dir_1.sort_values(by=[col_segment_id, col_distance], ascending=[True, False])
dir_1["ID"] = range(0, len(dir_1))
dir_1 = dir_1.set_index("ID")

dir = dir_0.append(dir_1)
dir = dir[[col_POI_ID, col_segment_id, col_directions, col_distance, col_type_ID, 'pois_type']]
dir = dir.sort_values(by=[col_segment_id, col_POI_ID, 'pois_type', col_distance])
dir = dir.drop_duplicates(subset=[col_POI_ID, col_directions])
dir["ID"] = range(0, len(dir))
dir = dir.set_index("ID")

n0 = len(dir_0)
n1 = len(dir_1)
k = len(dir)
n3 = k
# 50 kW annehmen -> + 20h durchgehende Besetzung
g = 100000  # Maximum number of charging poles at one charging station
specific_demand = 0.2  # (kWh/100km) average specific energy usage for 100km
acc = (
    specific_demand * 100
)  # (kWh) charged energy by a car during one charging for 100km
charging_capacity = 50  # (kW)
ec = 0.25  # (€/kWh) charging price for EV driver
e_tax = 0.15  # (€/kWh) total taxes and other charges
cfix = 150000  # (€) total installation costs of charging station installation
cvar = 10000  # (€) total installation costs of charging pole installation
eta = 0.011  # share of electric vehicles of car fleet
mu = 0.7  # share of cars travelling long-distance
hours_of_constant_charging = (
    20  # number of hours of continuous charging at one charging pole
)
energy_demand_0 = dir_0[
    col_energy_demand
].to_list()  # (kWh/d) energy demand at each rest area per day
energy_demand_1 = dir_1[col_energy_demand].to_list()
directions_0 = dir_0[col_directions].to_list()
directions_1 = dir_1[col_directions].to_list()

cars_per_day = hours_of_constant_charging / (acc / charging_capacity)
energy = acc * cars_per_day  # (kWh) charging energy per day by one charging pole

e_average = (
    (sum(energy_demand_0) + sum(energy_demand_1)) * eta * mu / (n0 + n1)
)  # (kWh/d) average energy demand at a charging station per day
i = 0.05  # interest rate
T = 10  # (a) calculation period für RBF (=annuity value)
RBF = 1 / i - 1 / (i * (1 + i) ** T)  # (€/a) annuity value for period T
dmax = 100000
# extracting all highway names to create two additional columns: "first" and "last" to indicate whether resting areas
# are first resting areas along a singular highway in "normal" direction

l1 = dir_1[col_segment_id].to_list()
l0 = dir_0[col_segment_id].to_list()
l_ext = l0
l_ext.extend(l1)
highway_names = list(set(l_ext))

# read existing infrastructure
ex_infr_0 = pd.read_csv("data/rest_areas_0_charging_stations.csv")
ex_infr_1 = pd.read_csv("data/rest_areas_1_charging_stations.csv")

# define masks
mask_0 = np.zeros((n0, n0))
mask_1 = np.zeros((n1, n1))
enum_0 = mask_0
enum_1 = mask_1
maximum_dist_between_charging_stations = 3000
energy_demand_matrix_0 = np.append(np.diag(energy_demand_0) * eta * mu, np.zeros([n0, n1]), axis=1)
energy_demand_matrix_1 = np.append(np.diag(energy_demand_1) * eta * mu, np.zeros([n1, n0]), axis=1)

all_indices_0 = dir_0.index.to_list()
all_indices_1 = dir_1.index.to_list()

for ij in range(0, len(dir_0)):
    current_highway = l0[ij]
    enum_0[ij, ij] = 1
    extract_dir_0_ind = dir_0[dir_0[col_segment_id] == current_highway].index.to_list()
    current_ind = all_indices_0[ij]
    count = 1
    for ind in extract_dir_0_ind:
        if ind > current_ind:
            count = count + 1
            enum_0[ij, ind] = count


for ij in range(0, len(dir_1)):
    current_highway = l1[ij]
    enum_1[ij, ij] = 1
    extract_dir_1_ind = dir_1[dir_1[col_segment_id] == current_highway].index.to_list()
    current_ind = all_indices_1[ij]
    count = 1
    for ind in extract_dir_1_ind:
        if ind > current_ind:
            count = count + 1
            enum_1[ij, ind] = count


mask_0 = np.where(enum_0 > 0, 1, 0)
mask_1 = np.where(enum_1 > 0, 1, 0)

