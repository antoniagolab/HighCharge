import pandas as pd
import numpy as np
import geopandas as gpd
import psycopg2 as ps
from shapely.ops import linemerge
import matplotlib.pyplot as plt
import re
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point, LineString

# connection to database containing all OSM data of Austria; downloaded from http://download.geofabrik.de/
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

# reference coordinate system for accurate calculations and estimations in metre
reference_coord_sys = "EPSG:31287"

# getting highways
highways = gpd.read_file('C:/Users/golab/Documents/ArcGIS/Projects/trafficflowproject/output_9/v21.shp')
highways = highways.to_crs(reference_coord_sys)    # transformation to MGI / Austria Lambert
highway_names = list(set(highways.ref.to_list()))
highway_names.sort()
highway_names_to_exclude = [None, 'S1']

# removing highways from the list of names which we do not want to include in further analysis
for name in highway_names_to_exclude:
    if name in highway_names:
        highway_names.remove(name)

# getting ASFINAG traffic flow information to retrieve normal and inverse directions of the
traffic_flow_data = pd.read_excel('C:/Users/golab\PycharmProjects/trafficflow/2020/2001_ASFINAG_Verkehrsstatistik_BW.xls', header=2)

# function for determining relative point position
def is_right(p1, p2, p3):
    pos_value = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    if pos_value < 0:
        return True
    else:
        return False

highway_geometry_list = []

for name in highway_names:
    # introducing here a double check for which of the linestrings of a MultiLineString geometry represents the lanes in
    # normal and which in inverse direction;
    # 1. Retrieval of starting point and ending point
    # adapting highway_name
    if len(name) < 3:
        name_asfinag = name[0] + str(0) + name[-1]
    else:
        name_asfinag = name

    if name in ['S1A', 'S1B']:
        name_asfinag = 'S01'
    traffic_flow_data_current_highway = traffic_flow_data[traffic_flow_data['Unnamed: 0'] == name_asfinag]

    # try:
    starting_point = traffic_flow_data_current_highway['Unnamed: 5'].to_list()[0]
    if '-' in starting_point:
        starting_point = starting_point.replace('-', ' ')
    elif ' ' in starting_point:
        starting_point = starting_point.split(' ')[-1]
    ending_point = traffic_flow_data_current_highway['Unnamed: 5'].to_list()[3]
    if name == 'S1B':
        ending_point = traffic_flow_data_current_highway['Unnamed: 5'].to_list()[-4]
    if '-' in ending_point:
        ending_point = ending_point.replace('-', ' ')
    elif ' ' in ending_point:
        ending_point = ending_point.split(' ')[-1]

    # search in points(3) + polygons(1) + line(2)
    # search in planet_osm_polygons
    starting_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where planet_osm_polygon.name LIKE '%" + starting_point + "%'", conn, geom_col="way", crs=3857)
    if len(starting_point_geom) == 0:
        # then search in planet_osm_line
        starting_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where planet_osm_line.name LIKE '%" + starting_point + "%'", conn, geom_col="way", crs=3857)
    # then search in planet_osm_point
    if len(starting_point_geom) == 0:
        starting_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_point where planet_osm_point.name LIKE '%" + starting_point + "%'", conn, geom_col="way", crs=3857)

    # search in planet_osm_polygon
    ending_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where planet_osm_polygon.name LIKE '%" + ending_point + "%'", conn, geom_col="way", crs=3857)
    if len(ending_point_geom) == 0:
        # then search in planet_osm_line
        ending_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where planet_osm_line.name LIKE '%" + ending_point + "%'", conn, geom_col="way", crs=3857)
    # then search in planet_osm_point
    if len(ending_point_geom) == 0:
        ending_point_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_point where planet_osm_point.name LIKE '%" + ending_point + "%'", conn, geom_col="way", crs=3857)

    starting_point_geom_final = starting_point_geom.to_crs(reference_coord_sys).geometry.to_list()[0]
    ending_point_geom_final = ending_point_geom.to_crs(reference_coord_sys).geometry.to_list()[0]
    # extracting points for each of the LineStrings representing both directions
    highway_geom = highways[highways.ref == name].geometry.to_list()[0]
    l1 = highway_geom[0]
    l2 = highway_geom[1]
    l1_points = [Point(l1.xy[0][kl], l1.xy[1][kl]) for kl in range(0, len(l1.xy[0]))]
    l2_points = [Point(l2.xy[0][kl], l2.xy[1][kl]) for kl in range(0, len(l2.xy[0]))]

    # finding nearest points to starting point
    point_1_1 = l1_points[0]
    point_1_2 = l1_points[-1]

    point_2_1 = l2_points[0]
    point_2_2 = l2_points[-1]

    if point_1_1.distance(starting_point_geom_final) > point_1_2.distance(starting_point_geom_final):
        point_1 = point_1_1
        point_3 = point_1_2
    else:
        point_1 = point_1_2
        point_3 = point_1_1

    if point_2_1.distance(starting_point_geom_final) > point_2_2.distance(starting_point_geom_final):
        point_2 = point_2_1
        point_4 = point_2_2
    else:
        point_2 = point_2_2
        point_4 = point_2_1

    # calculating point right in between
    middle_point_start = Point((point_1.x + point_2.x)/2, (point_1.y + point_2.y)/2)
    middle_point_end = Point((point_3.x + point_4.x)/2, (point_3.y + point_4.y)/2)

    # determining left and right points of middle point given the vector symbolizing highway normal direction
    geom_ind_normal_dir = None
    geom_ind_inverse_dir = None

    if is_right(middle_point_start, middle_point_end, point_1):
        geom_ind_normal_dir = 0
    if is_right(middle_point_start, middle_point_end, point_2):
        geom_ind_normal_dir = 1
    if not is_right(middle_point_start, middle_point_end, point_1):
        geom_ind_inverse_dir = 0
    if not is_right(middle_point_start, middle_point_end, point_2):
        geom_ind_inverse_dir = 1

    dir_list = [geom_ind_normal_dir, geom_ind_inverse_dir]
    dir_list.sort()

    # printing warning if algorithm does something totally wrong
    if not dir_list == [0, 1]:
        print("WARNING: For highway " + name + " happened an impossible matching!")

    # now sorting points of each LineString according to driving normal direction

    # line 1
    if point_1 == point_1_1:
        final_line_1_geom = l1
    else:
        l1_points.reverse()
        final_line_1_geom = LineString(l1_points)

    # line 2
    if point_2 == point_2_1:
        final_line_2_geom = l2
    else:
        l2_points.reverse()
        final_line_2_geom = LineString(l2_points)

    if geom_ind_normal_dir == 0 and geom_ind_inverse_dir == 1:
        highway_geometry_list.append({'highway': name, 'normal': True, 'geometry': final_line_1_geom.wkt})
        highway_geometry_list.append({'highway': name, 'normal': False, 'geometry': final_line_2_geom.wkt})
    elif geom_ind_normal_dir == 1 and geom_ind_inverse_dir == 0:
        highway_geometry_list.append({'highway': name, 'normal': True, 'geometry': final_line_2_geom.wkt})
        highway_geometry_list.append({'highway': name, 'normal': False, 'geometry': final_line_1_geom.wkt})
    else:
        print("WARNING: Not added: " + name)
    print(name + " processed")

    #except:
    #print(name + " did not work")
    #continue

highway_geometry_df = pd.DataFrame(highway_geometry_list)
highway_geometry_df.to_csv('geometries/highway_geometries_3.csv')