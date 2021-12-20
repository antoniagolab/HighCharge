import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiLineString
import psycopg2 as ps
import numpy as np

# database connection
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

# reference coordinate system for all calculations
reference_coord_sys = "EPSG:31287"

# highways
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v5.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_names = list(set(highway_geometries.highway.to_list()))
highway_names.sort()

# rest areas
rest_areas_0 = pd.read_csv(r'data/reviewed_resting_areas_0.csv')
rest_areas_1 = pd.read_csv(r'data/reviewed_resting_araes_1.csv')

# get service areas from DB
service_stations_polygons = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where highway='services' "                                       
                                                          "or highway='rest_area'", conn, geom_col="way", crs=3857)
service_stations_lines = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where highway='services' or "
                                                       "highway='rest_area'", conn, geom_col="way", crs=3857)

# buffer to draw around evaluated point along highway to search for OSM
buffer_around_pos = 1.5 * 1000   # 2 km radius

# buffer around line
buffer_highway = 300


# function for determining relative point position
def is_right(p1, p2, p3):
    pos_value = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    if pos_value < 0:
        return True
    else:
        return False


rest_area_info = []

for name in ['S1A', 'S1B']:
    # highway geometry
    extract_highway_geom = highway_geometries[highway_geometries.highway == name]
    line_0 = extract_highway_geom[extract_highway_geom.normal].geometry.to_list()[0]
    line_1 = extract_highway_geom[~extract_highway_geom.normal].geometry.to_list()[0]

    service_stations_polygons = service_stations_polygons.to_crs(reference_coord_sys)
    service_stations_lines = service_stations_lines.to_crs(reference_coord_sys)

    if len(name) == 2:
        asfinag_name = "'" + name[0] + '0' + name[-1] + "'"
    else:
        asfinag_name = "'" + name + "'"
    if name in ['S1A', 'S1B']:
        name_asfinag = "'S01'"

    extract_resting_areas_dir_0 = rest_areas_0[rest_areas_0.Autobahn == asfinag_name]
    extract_resting_areas_dir_1 = rest_areas_1[rest_areas_1.Autobahn == asfinag_name]
    extract_resting_areas_dir_0 = extract_resting_areas_dir_0.sort_values(by=['Standort'])
    extract_resting_areas_dir_1 = extract_resting_areas_dir_1.sort_values(by=['Standort'])
    extract_resting_areas = extract_resting_areas_dir_0.append(extract_resting_areas_dir_1)
    extract_resting_areas = extract_resting_areas.drop_duplicates(subset=['Name'])
    extract_resting_areas = extract_resting_areas.sort_values(by=['Standort'])

    buffer_around_highway = MultiLineString([line_0, line_1]).buffer(buffer_highway)

    ss_in_buffer_poly = service_stations_polygons[service_stations_polygons.intersects(buffer_around_highway)]
    ss_in_buffer_lines = service_stations_lines[service_stations_lines.intersects(buffer_around_highway)]
    ss_osm = ss_in_buffer_lines.append(ss_in_buffer_poly)
    ss_osm['centroid'] = ss_osm.geometry.centroid
    ss_osm['dist_along_highway'] = [line_0.project(p) for p in ss_osm['centroid'].to_list()]
    ss_osm = ss_osm.sort_values(by=['dist_along_highway'])
    count_ra = len(extract_resting_areas)
    counter = 0
    while counter < count_ra and counter < len(ss_osm):
        rest_area_info.append({'highway': name, 'name': extract_resting_areas.Name.to_list()[counter],
                               'asfinag_position': extract_resting_areas.Standort.to_list()[counter],
                               'asfinag_type': extract_resting_areas.Art.to_list()[counter],
                               'direction': extract_resting_areas.Richtung.to_list()[counter],
                               'osm_id': ss_osm.osm_id.to_list()[counter],
                               'osm_name': ss_osm.name.to_list()[counter]})
        counter = counter + 1

    while counter < count_ra:
        rest_area_info.append({'highway': name, 'name': extract_resting_areas.Name.to_list()[counter],
                               'asfinag_position': extract_resting_areas.Standort.to_list()[counter],
                               'asfinag_type': extract_resting_areas.Art.to_list()[counter],
                               'direction': extract_resting_areas.Richtung.to_list()[counter],
                               'osm_id': None, 'osm_name': None})
        counter = counter + 1
    while counter < len(ss_osm):
        rest_area_info.append({'highway': name, 'name': None, 'asfinag_position': None, 'asfinag_type': None,
                               'direction': 999, 'osm_id': ss_osm.osm_id.to_list()[counter],
                               'osm_name': ss_osm.name.to_list()[counter]})
        counter = counter + 1

rest_area_info = pd.DataFrame(rest_area_info)
rest_area_info_0 = rest_area_info[rest_area_info.direction.isin([0, 2])]
rest_area_info_1 = rest_area_info[rest_area_info.direction.isin([1, 2, 999])]

rest_area_info_0.to_csv('data/rest_area_info_0_s1.csv')
rest_area_info_1.to_csv('data/rest_area_info_1_s1.csv')

    # if len(name) < 3:
    #     asfinag_name = "'" + name[0] + str(0) + name[-1] + "'"
    # else:
    #     asfinag_name = "'" + name + "'"
    # extract_ra_0 = rest_areas_0[rest_areas_0.Autobahn == asfinag_name]
    # extract_ra_1 = rest_areas_1[rest_areas_1.Autobahn == asfinag_name]
    # # in dir 0
    # if len(extract_ra_0) > 0:
    #     extract_ra_2 = extract_ra_0[extract_ra_0.Richtung == 2]
    #     if len(extract_ra_2) > 0:
    #         for ij in range(0, len(extract_ra_2)):
    #             ra_name = extract_ra_2.Name.to_list()[ij]
    #             dist = extract_ra_2.Standort.to_list()[ij] * 1000
    #             pos_ra = line_0.interpolate(dist)
    #             pos_buffer = pos_ra.buffer(buffer_around_pos)
    #
    #             # search in radius in OSM data
    #             possible_geoms = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where st_contains(st_geomfromtext('" + pos_buffer.wkt +"'," + reference_coord_sys.split(":")[-1] + ")" +",st_transform(way, "+ reference_coord_sys.split(':')[-1] + ")) and( highway='services' or highway='rest_area')", conn, geom_col="way", crs=3857)
    #             if len(possible_geoms) == 0:
    #                 possible_geoms = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where st_contains(st_geomfromtext('" + pos_buffer.wkt +"'," + reference_coord_sys.split(":")[-1] + ")" +",st_transform(way, "+ reference_coord_sys.split(':')[-1] + ")) and (highway='services' or highway='rest_area')", conn, geom_col="way", crs=3857)
    #
    #             possible_geoms = possible_geoms.to_crs(reference_coord_sys)
    #
    #             # if multiple geometries lie within the buffer, the closest one is chosen
    #             if len(possible_geoms) > 1:
    #                 # insert here placing something at the index; name + insert osm_id + geometry + projected centroid
    #                 # insert warning if len(possible_geoms) > 0
    #                 dists = []
    #                 for kl in range(0, len(possible_geoms)):
    #                     current_geom = possible_geoms.geometry.to_list()[kl]
    #                     dists.append(current_geom.distance(pos_ra))
    #                 ind_min_dist = np.argmin(dists)
    #                 possible_geoms = possible_geoms[possible_geoms.index == ind_min_dist]
    #
    #             if len(possible_geoms) == 1:
    #                 ind_ra_0 = rest_areas_0[rest_areas_0.Name == ra_name].index.to_list()[0]
    #                 rest_areas_0.at[ind_ra_0, 'osm_id'] = possible_geoms.osm_id.to_list()[0]
    #                 try:
    #                     rest_areas_0.at[ind_ra_0, 'osm_name'] = possible_geoms.name.to_list()[0]
    #                 except:
    #                     print("Something wrong with rest area :" + ra_name +"; " + name)
    #                     continue
    #                 rest_areas_0.at[ind_ra_0, 'geometry'] = possible_geoms.geometry.to_list()[0].wkt
    #                 rest_areas_0.at[ind_ra_0, 'centroid'] = possible_geoms.geometry.to_list()[0].centroid.wkt
    #                 rest_areas_0.at[ind_ra_0, 'centroid_dist'] = line_0.project(possible_geoms.geometry.to_list()[0].centroid)
    #
    #                 ind_ra_1 = rest_areas_1[rest_areas_1.Name == ra_name].index.to_list()[0]
    #                 rest_areas_1.at[ind_ra_1, 'osm_id'] = possible_geoms.osm_id.to_list()[0]
    #                 try:
    #                     rest_areas_1.at[ind_ra_1, 'osm_name'] = possible_geoms.name.to_list()[0]
    #                 except:
    #                     print("Something wrong with rest area :" + ra_name +"; " + name)
    #                 rest_areas_1.at[ind_ra_1, 'geometry'] = possible_geoms.geometry.to_list()[0].wkt
    #                 rest_areas_1.at[ind_ra_1, 'centroid'] = possible_geoms.geometry.to_list()[0].centroid.wkt
    #                 rest_areas_1.at[ind_ra_1, 'centroid_dist'] = line_0.project(
    #                     possible_geoms.geometry.to_list()[0].centroid)
    #
    # extract_0_1 = extract_ra_0.append(extract_ra_1)
    # extract_0_1 = extract_0_1[~extract_0_1.Richtung.isin([2])]
    # for ij in range(0, len(extract_0_1)):
    #     ra_name = extract_0_1.Name.to_list()[ij]
    #     dist = extract_0_1.Standort.to_list()[ij] * 1000
    #     dir = extract_0_1.Richtung.to_list()[ij]
    #     pos_ra = line_0.interpolate(dist)
    #     pos_ra_plus_20 = line_0.interpolate(dist+20)
    #     pos_buffer = pos_ra.buffer(buffer_around_pos)
    #
    #     # search in radius in OSM data
    #     possible_geoms = gpd.GeoDataFrame.from_postgis(
    #         "select * from planet_osm_line where st_intersects(st_geomfromtext('" + pos_buffer.wkt + "'," +
    #         reference_coord_sys.split(":")[-1] + ")" + ",st_transform(way, " + reference_coord_sys.split(':')[
    #             -1] + ")) and( highway='services' or highway='rest_area')", conn, geom_col="way", crs=3857)
    #     if len(possible_geoms) == 0:
    #         possible_geoms = gpd.GeoDataFrame.from_postgis(
    #             "select * from planet_osm_polygon where st_intersects(st_geomfromtext('" + pos_buffer.wkt + "'," +
    #             reference_coord_sys.split(":")[-1] + ")" + ",st_transform(way, " + reference_coord_sys.split(':')[
    #                 -1] + ")) and (highway='services' or highway='rest_area')", conn, geom_col="way", crs=3857)
    #
    #     possible_geoms = possible_geoms.to_crs(reference_coord_sys)
    #
    #     # if multiple geometries lie within the buffer, the closest one is chosen
    #     if len(possible_geoms) > 1:
    #         # first take only the ones which are right(0) or left(1) of the line
    #         is_right_list = []
    #         for kl in range(0, len(possible_geoms)):
    #             current_geom = possible_geoms.geometry.to_list()[kl]
    #             centroid_of_geom = current_geom.centroid
    #             is_right_list.append(is_right(pos_ra, pos_ra_plus_20, centroid_of_geom))
    #         possible_geoms['is_right'] = is_right_list
    #         if dir == 0:
    #             possible_geoms = possible_geoms[possible_geoms['is_right']]
    #         else:
    #             possible_geoms = possible_geoms[~possible_geoms['is_right']]
    #
    #         if len(possible_geoms) > 1:
    #             dists = []
    #             for kl in range(0, len(possible_geoms)):
    #                 current_geom = possible_geoms.geometry.to_list()[kl]
    #                 dists.append(current_geom.distance(pos_ra))
    #             ind_min_dist = np.argmin(dists)
    #             possible_geoms = possible_geoms[possible_geoms.index == ind_min_dist]
    #     if len(possible_geoms) > 0:
    #         if dir == 0:
    #             ind_rest_0 = rest_areas_0[rest_areas_0.Name == ra_name]
    #             rest_areas_0.at[ind_ra_0, 'osm_id'] = possible_geoms.osm_id.to_list()[0]
    #             try:
    #                 rest_areas_0.at[ind_ra_0, 'osm_name'] = possible_geoms.name.to_list()[0]
    #             except:
    #                 print("Something wrong with rest area :" + ra_name + "; " + name)
    #             rest_areas_0.at[ind_ra_0, 'geometry'] = possible_geoms.geometry.to_list()[0].wkt
    #             rest_areas_0.at[ind_ra_0, 'centroid'] = possible_geoms.geometry.to_list()[0].centroid.wkt
    #             rest_areas_0.at[ind_ra_0, 'centroid_dist'] = line_0.project(
    #                 possible_geoms.geometry.to_list()[0].centroid)
    #         if dir == 1:
    #             ind_rest_1 = rest_areas_1[rest_areas_1.Name == ra_name]
    #             rest_areas_1.at[ind_ra_1, 'osm_id'] = possible_geoms.osm_id.to_list()[0]
    #             try:
    #                 rest_areas_1.at[ind_ra_1, 'osm_name'] = possible_geoms.name.to_list()[0]
    #             except:
    #                 print("Something wrong with rest area :" + ra_name + "; " + name)
    #             rest_areas_1.at[ind_ra_1, 'geometry'] = possible_geoms.geometry.to_list()[0].wkt
    #             rest_areas_1.at[ind_ra_1, 'centroid'] = possible_geoms.geometry.to_list()[0].centroid.wkt
    #             rest_areas_1.at[ind_ra_1, 'centroid_dist'] = line_0.project(
    #                 possible_geoms.geometry.to_list()[0].centroid)

