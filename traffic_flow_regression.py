import geopandas as gpd
import pandas as pd
import math
import numpy as np
from neupy import algorithms
from edit_asfinag_data import *
import matplotlib.pyplot as plt

# get traffic counter positions
tc = pd.read_csv('data/traffic_counters_positions_v23.csv')

# get traffic flow data
filename = 'data/Verkehrsstatistik Juni 2021'
edit_asfinag_file(filename)
asfinag_data = pd.read_csv(filename + "_edited.csv")



# filtering asfinag data after vehicle types of interest
vehicle_type_of_interest = 'Kfz <= 3,5t hzG'
filtered_asfinag_data = asfinag_data[asfinag_data['Unnamed: 6'] == vehicle_type_of_interest]

# collecting highway names
highway_names = list(set(tc.highway.to_list()))

# normal and inverse direction
tc_0 = tc
tc_1 = tc

# match for each direction
normal_directions = []
inverse_directions = []
for name in highway_names:
    if len(name) < 3:
        asfinag_name = name[0] + str(0) + name[-1]
    else:
        asfinag_name = name
    extract_asfinag_data = filtered_asfinag_data[filtered_asfinag_data['Unnamed: 0'] == asfinag_name]
    normal_directions.append(extract_asfinag_data['Unnamed: 5'].to_list()[0])
    inverse_directions.append(extract_asfinag_data['Unnamed: 5'].to_list()[1])

traffic_flow_data_0 = filtered_asfinag_data[filtered_asfinag_data['Unnamed: 5'].isin(normal_directions)]
traffic_flow_data_1 = filtered_asfinag_data[filtered_asfinag_data['Unnamed: 5'].isin(inverse_directions)]

traffic_flow_data_0['nb'] = traffic_flow_data_0['Unnamed: 3']
traffic_flow_data_1['nb'] = traffic_flow_data_1['Unnamed: 3']

traffic_flow_data_0['traffic_flow'] = traffic_flow_data_0['Kfz/24h']
traffic_flow_data_1['traffic_flow'] = traffic_flow_data_1['Kfz/24h']

traffic_flow_data_0 = traffic_flow_data_0[['nb', 'traffic_flow']]
traffic_flow_data_1 = traffic_flow_data_1[['nb', 'traffic_flow']]

# merge
tc_flow_0 = pd.merge(tc_0, traffic_flow_data_0, on='nb')
tc_flow_1 = pd.merge(tc_1, traffic_flow_data_1, on='nb')

# delete all traffic counters where value == -1 as these are not actual values usable for the regression
tc_flow_0 = tc_flow_0[~(tc_flow_0.traffic_flow == -1)]
tc_flow_1 = tc_flow_1[~(tc_flow_1.traffic_flow == -1)]
# get rest areas
rest_areas_0 = pd.read_csv('data/adapted_rest_areas_0.csv')
rest_areas_1 = pd.read_csv('data/adapted_rest_areas_1.csv')

# standard deviation
std = 1000


# make GRNN
for name in highway_names:
    # normal direction
    extract_tc_0 = tc_flow_0[tc_flow_0.highway == name]
    extract_tc_0 = extract_tc_0.sort_values(by='position')
    nw = algorithms.GRNN(std=std)
    nw.train(np.array(extract_tc_0.position), np.array(extract_tc_0.traffic_flow))
    extract_ra_0 = rest_areas_0[rest_areas_0.highway == name]
    pred_traffic_flow_0 = nw.predict(np.array(extract_ra_0.dist_along_highway))
    rest_areas_0.at[extract_ra_0.index, 'traffic_flow'] = pred_traffic_flow_0
    plt.figure()
    plt.plot(extract_ra_0.dist_along_highway.to_list(), pred_traffic_flow_0, '*-', label='estimated traffic counts at rest areas')
    plt.plot(np.array(extract_tc_0.position), np.array(extract_tc_0.traffic_flow), '*-', label='traffic count by ASFINAG')
    plt.xlabel('distance (km)')
    plt.ylabel('traffic count')
    plt.title(name + ' dir=0')
    plt.legend()
    plt.xlim([min(extract_ra_0.dist_along_highway.to_list() + extract_tc_0.position.to_list()), max(extract_ra_0.dist_along_highway.to_list() + extract_tc_0.position.to_list()) + 5000])
    plt.savefig('traffic_counts/' + name + '_dir_0.png')


    # inverse direction
    extract_tc_1 = tc_flow_1[tc_flow_1.highway == name]
    extract_tc_1 = extract_tc_1.sort_values(by='position')
    nw = algorithms.GRNN(std=std)
    nw.train(np.array(extract_tc_1.position), np.array(extract_tc_1.traffic_flow))
    extract_ra_1 = rest_areas_1[rest_areas_1.highway == name]
    pred_traffic_flow_1 = nw.predict(np.array(extract_ra_1.dist_along_highway))
    rest_areas_1.at[extract_ra_1.index, 'traffic_flow'] = pred_traffic_flow_1
    plt.figure()
    plt.plot(extract_ra_1.dist_along_highway.to_list(), pred_traffic_flow_1, '*-', label='estimated traffic counts at rest areas')
    plt.plot(np.array(extract_tc_1.position), np.array(extract_tc_1.traffic_flow), '*-', label='traffic count by ASFINAG')
    plt.xlabel('distance (km)')
    plt.ylabel('traffic count')
    plt.xlim([min(extract_ra_1.dist_along_highway.to_list() + extract_tc_1.position.to_list()), max(extract_ra_1.dist_along_highway.to_list() + extract_tc_1.position.to_list()) + 5000])
    plt.title(name + ' dir=1')
    plt.legend()
    plt.savefig('traffic_counts/' + name + '_dir_1.png')
#
rest_areas_0.to_csv('data/rest_areas_0_with_traffic_flow_v2.csv', index=False)
rest_areas_1.to_csv('data/rest_areas_1_with_traffic_flow_v2.csv', index=False)