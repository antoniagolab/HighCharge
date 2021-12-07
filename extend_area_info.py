import pandas as pd
import geopandas as gpd
import psycopg2 as ps
from shapely import wkt
from edit_asfinag_data import *

# setting up database connection
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

reference_coord_sys = "EPSG:31287"


# highways
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v6.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_names = list(set(highway_geometries.highway.to_list()))
highway_names.sort()


# locating by averaging to positions: (1) intersection between circle with radius of position-diff and highway geometry
# (2) interpolated position_diff
# which is a compromise between both possibilities


# rest areas
rest_areas_0 = pd.read_csv(r'data/adapted_rest_areas_0.csv')
rest_areas_1 = pd.read_csv(r'data/adapted_rest_areas_1.csv')

traffic_counter_info = []

# calculate centroids and project onto line geometries
osm_ids = rest_areas_0.osm_id.to_list() + rest_areas_1.osm_id.to_list()
highways_list = rest_areas_0.highway.to_list() + rest_areas_1.highway.to_list()

for ij in range(0, len(osm_ids)):
    osm_id = osm_ids[ij]
    current_highway_name = highways_list[ij]
    highway_geom = highway_geometries[highway_geometries.highway == current_highway_name].geometry.to_list()[0]
    ra_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_polygon where osm_id=" + str(osm_id), conn, geom_col="way", crs=3857)
    if len(ra_geom) == 0:
        ra_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where osm_id=" + str(osm_id), conn, geom_col="way", crs=3857)
    if len(ra_geom) == 0:
        ra_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_point where osm_id=" + str(osm_id), conn, geom_col="way", crs=3857)
    if len(ra_geom) == 0:
        ra_geom = gpd.GeoDataFrame.from_postgis("select * from planet_osm_roads where osm_id=" + str(osm_id), conn, geom_col="way", crs=3857)

    ra_geom = ra_geom.to_crs(reference_coord_sys).geometry.to_list()[0]

    if ij < len(rest_areas_0):
        ind_ras = rest_areas_0[rest_areas_0.osm_id == osm_id].index.to_list()
        for ind in ind_ras:
            rest_areas_0.at[ind, 'centroid'] = ra_geom.centroid.wkt
            rest_areas_0.at[ind, 'dist_along_highway'] = highway_geom.project(ra_geom.centroid)
            print(str(current_highway_name) + "worked")
    else:
        ind_ras = rest_areas_1[rest_areas_1.osm_id == osm_id].index.to_list()
        for ind in ind_ras:
            rest_areas_1.at[ind, 'centroid'] = ra_geom.centroid.wkt
            rest_areas_1.at[ind, 'dist_along_highway'] = highway_geom.project(ra_geom.centroid)
            print(str(current_highway_name) + "worked")


rest_areas_0.to_csv('data/rest_areas_with_centroids_0.csv', index=False)
rest_areas_1.to_csv('data/rest_areas_with_centroids_1.csv', index=False)