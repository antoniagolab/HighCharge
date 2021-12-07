import numpy as np
import pandas as pd
import geopandas as gpd
import psycopg2 as ps
from shapely import wkt
from edit_asfinag_data import *

# setting up database connection
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

# reference coordinate system for all calculations
reference_coord_sys = "EPSG:31287"


# highways
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v6.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_names = list(set(highway_geometries.highway.to_list()))
highway_names.sort()

highways_to_el = [None, 'A23']
for el in highways_to_el:
    if el in highway_names:
        highway_names.remove(el)


# rest areas
rest_areas_0 = pd.read_csv(r'data/adapted_rest_areas_0.csv')
rest_areas_1 = pd.read_csv(r'data/adapted_rest_areas_1.csv')

# motorway junctions
motorway_junctions = pd.read_csv(r'data/motorway_junctions.csv')

# vehicle count by ASFINAG
filename = '2020/2001_ASFINAG_Verkehrsstatistik_BW'
edit_asfinag_file(filename)
asfinag_data = pd.read_csv(filename + "_edited.csv")

# filtering asfinag data after vehicle types of interest
vehicle_type_of_interest = 'Kfz <= 3,5t hzG'
filtered_asfinag_data = asfinag_data[asfinag_data['Unnamed: 6'] == vehicle_type_of_interest]

traffic_counter_info = []

# calculate centroids and project onto line geometries
osm_ids = rest_areas_0.osm_id.to_list() + rest_areas_1.osm_id.to_list()
highways_list = rest_areas_0.highway.to_list() + rest_areas_1.highway.to_list()

dists_along_highway = rest_areas_0.dist_along_highway.to_list() + rest_areas_1.dist_along_highway.to_list()
asfinag_positions = rest_areas_0.asfinag_position.to_list() + rest_areas_1.asfinag_position.to_list()
rest_area_collection = rest_areas_0.append(rest_areas_1)
indices_ra_collection = rest_area_collection.index.to_list()
ra_centroids = rest_area_collection.centroid.to_list()
traffic_counter_info = []

s1a_length = 15965.337629673604

for name in highway_names:
    if len(name) < 3:
        asfinag_name = name[0] + str(0) + name[-1]
    else:
        asfinag_name = name

    extract_rest_areas_0 = rest_areas_0[rest_areas_0.highway == name]
    extract_rest_areas_1 = rest_areas_1[rest_areas_1.highway == name]
    extract_motorway_junctions = motorway_junctions[motorway_junctions.highway == name]
    extract_rest_areas = extract_rest_areas_0.append(extract_rest_areas_1)
    extract_rest_areas['geometry'] = extract_rest_areas['centroid']
    check_points_along_highway = extract_rest_areas.append(extract_motorway_junctions)
    check_points_along_highway = check_points_along_highway.drop_duplicates(subset=['name', 'direction'])
    check_points_along_highway['asfinag_position'] = [float(d) for d in check_points_along_highway.asfinag_position.to_list()]
    check_points_along_highway = check_points_along_highway.sort_values(by=['asfinag_position'])

    ra_dists_along_highway = check_points_along_highway.dist_along_highway.to_list()
    ra_asfinag_positions = check_points_along_highway.asfinag_position.to_list()
    ra_indices = check_points_along_highway.index.to_list()
    ra_centroids = check_points_along_highway.geometry.to_list()
    ra_names = check_points_along_highway.name.to_list()

    extract_asfinag_data = filtered_asfinag_data[filtered_asfinag_data['Unnamed: 0'] == asfinag_name]

    highway_geom = highway_geometries[highway_geometries.highway == name].geometry.to_list()[0]
    # collecting traffic flow counter references
    traffic_counter_numbers = list(set(extract_asfinag_data['Unnamed: 3'].to_list()))
    for kl in range(0, len(traffic_counter_numbers)):
        dist = extract_asfinag_data[extract_asfinag_data['Unnamed: 3'] == traffic_counter_numbers[kl]]['[km]'].to_list()[0]
        # find closest rest area
        diff_distances = [(dist - n) for n in ra_asfinag_positions]
        abs_diff_distances = [np.absolute(d) for d in diff_distances]
        ind_min_diff = np.argmin(abs_diff_distances)
        index_closest_ra = ra_indices[ind_min_diff]
        real_dist_on_highway = ra_dists_along_highway[ind_min_diff]
        ra_position = wkt.loads(ra_centroids[ind_min_diff])
        ra_before = True
        if diff_distances[ind_min_diff] < 0:
            ra_before = False
        projected_ra_centroid = highway_geom.interpolate(highway_geom.project(ra_position))
        buffer_radius = abs_diff_distances[ind_min_diff] * 1000
        possible_tc_positions = highway_geom.intersection(projected_ra_centroid.buffer(buffer_radius).exterior)
        if possible_tc_positions.type == 'MultiPoint':
            dists_intersections_along_highway = [highway_geom.project(p) for p in possible_tc_positions]
            diffs_to_ra_position = [real_dist_on_highway-n for n in dists_intersections_along_highway]
            if not ra_before:
                diffs_to_ra_position = [(-1) * d for d in diffs_to_ra_position]
            for mn in range(0, len(diffs_to_ra_position)):
                if diffs_to_ra_position[mn] > 0:
                    diffs_to_ra_position[mn] = 999999
                else:
                    diffs_to_ra_position[mn] = diffs_to_ra_position[mn]

            ind_belong_intersection = np.argmin(diffs_to_ra_position)
            tc_position = possible_tc_positions[ind_belong_intersection]
        elif possible_tc_positions.type == 'Point':
            tc_position = possible_tc_positions
        else:
            print("something wrong!")
            tc_position = None

        if buffer_radius < 200:
            traffic_counter_info.append({'nb': traffic_counter_numbers[kl], 'highway': name,
                                         'position': highway_geom.project(tc_position), 'geometry': tc_position.wkt,
                                         'check_needed': True})
        else:
            traffic_counter_info.append({'nb': traffic_counter_numbers[kl], 'highway': name,
                                         'position': highway_geom.project(tc_position), 'geometry': tc_position.wkt,
                                         'check_needed': False})
    print(name + " fully processed")

traffic_counter_info = pd.DataFrame(traffic_counter_info)
traffic_counter_info.to_csv('data/traffic_counters_v24.csv', index=False)
traffic_counter_info['geometry'] = traffic_counter_info.geometry.apply(wkt.loads)
traffic_counter_info_gpd = gpd.GeoDataFrame(traffic_counter_info, crs=reference_coord_sys, geometry='geometry')

traffic_counter_info_gpd.to_file('geometries/tc_v24.shp')







