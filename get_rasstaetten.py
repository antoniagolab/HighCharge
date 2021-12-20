import pandas as pd
import numpy as np
import geopandas as gpd
import psycopg2 as ps
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point, LineString


# TODO Limitations to correct in future
#   - automatic detect of directions for a rest/service area
#

reference_coord_sys = "EPSG:31287"
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

# get raststaetten

service_stations_polygons = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where highway='services' "
                                                          "or highway='rest_area'", conn, geom_col="way", crs=3857)
service_stations_lines = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where highway='services' or "
                                                       "highway='rest_area'", conn, geom_col="way", crs=3857)
service_stations_polygons = service_stations_polygons.to_crs(reference_coord_sys)
service_stations_lines = service_stations_lines.to_crs(reference_coord_sys)
highways = gpd.read_file('C:/Users/golab/Documents/ArcGIS/Projects/trafficflowproject/output_10/v16.shp')

highways = highways.to_crs(reference_coord_sys)    # transformation to MGI / Austria Lambert
# for further reference: https://secure.umweltbundesamt.at/edm_portal/redaList.do?d-49520-s=1&d-49520-p=1&seqCode=7cjpzbfdqa5cka&lsCode=qijtpm52ffdip4&display=plain&d-49520-o=1

# reading old data set
resting_areas_dir_0 = pd.read_csv(r'C:\Users\golab\PycharmProjects/trafficflow\data/resting_areas_dir_0.csv')
resting_areas_dir_1 = pd.read_csv(r'C:\Users\golab\PycharmProjects/trafficflow\data/resting_areas_dir_1.csv')
old_data_highway_names = list(set(resting_areas_dir_0.Autobahn.to_list() + resting_areas_dir_1.Autobahn.to_list()))
old_data_highway_names.sort()

buffer_size = 300

# Probelauf mit diesem highway
highway_names = list(set(highways.ref.to_list()))
if None in highway_names:
    highway_names.remove(None)
highway_names.sort()
highways_to_check = []
for name in highway_names:
    if len(name) == 2:
        current_highway_name = "'" + name[0] + '0' + name[-1]+ "'"
    else:
        current_highway_name = "'" + name + "'"
    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    total_old_count = 0
    if current_highway_name in old_data_highway_names:
        extract_resting_areas_dir_0 = resting_areas_dir_0[resting_areas_dir_0.Autobahn == current_highway_name]
        extract_resting_areas_dir_1 = resting_areas_dir_1[resting_areas_dir_1.Autobahn == current_highway_name]
        counter_2 = len(extract_resting_areas_dir_0[extract_resting_areas_dir_0.Richtung == 2])
        counter_0 = len(extract_resting_areas_dir_0[extract_resting_areas_dir_0.Richtung == 0])
        counter_1 = len(extract_resting_areas_dir_1[extract_resting_areas_dir_1.Richtung == 1])
        total_old_count = counter_0 + counter_1 + counter_2

    highway_geom = highways[highways.ref == name].geometry
    buffer_around_highway = highway_geom.buffer(buffer_size).to_list()[0]
    ss_in_buffer_poly = service_stations_polygons[service_stations_polygons.intersects(buffer_around_highway)]
    ss_in_buffer_lines = service_stations_lines[service_stations_lines.intersects(buffer_around_highway)]

    total_new_count = len(ss_in_buffer_poly) + len(ss_in_buffer_lines)
    if not total_new_count == total_old_count:
        print(str(name) + "\n" + "Old: " + str(total_old_count) + "\n New: " + str(total_new_count))
        continue


# identifying abnormal geometries - standard: 2

for name in highway_names:
    currently_checked_geom = highways[highways.ref == name].geometry
    if not len(currently_checked_geom.to_list()[0]) == 2:
        ax = currently_checked_geom.plot()
        ax.set_title(name)
# ----------------------------------------------------------------------------------------------------------------------

#
# # TODO: UPDATE rest area data
# # step 1: identify missing rest areas in current data set from 2007 -check
# # step 2: complete these - check
# # step 3: identify abnormal highway geometries which need a second look -> Ueberarbeitung check
# # step 4: identify absolute geometries of these (+osm_id, + way/line)
# # step 5:
#
