import numpy as np
import pandas as pd
import geopandas as gpd
import psycopg2 as ps
from shapely import wkt
from shapely.geometry import Point, MultiLineString


# setting up database connection
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

highways_to_el = [None]
for el in highways_to_el:
    if el in highway_names:
        highways_to_el.remove(el)

buffer_around_highway = 100
motorway_junction_info = pd.DataFrame()

for name in ['A9']:
    highway_geom = MultiLineString(highway_geometries[highway_geometries.highway == name].geometry.to_list())
    junctions = gpd.GeoDataFrame.from_postgis("select * from planet_osm_point where st_contains(st_geomfromtext('"
                                  + highway_geom.buffer(buffer_around_highway).wkt + "',"
                                  + reference_coord_sys.split(':')[-1] + "),st_transform(way,"
                                  + reference_coord_sys.split(':')[-1] + ")) and highway='motorway_junction' and name not like any (array['%Rastplatz%', '%Raststation%', '%Parkplatz%'])",
                                  conn, geom_col="way", crs=3857)
    junctions = junctions[~junctions.ref.isna()]
    junctions = junctions.to_crs(reference_coord_sys)
    # for each junction -> add asfinag pos, dist along highway, geometry
    # first add dist along highway
    # sort along
    motorway_junction_info_for_highway = []
    for ij in range(0, len(junctions)):
        junction_geom = junctions.way.to_list()[ij]
        junction_name = junctions.name.to_list()[ij]
        if 'Raststation' not in junction_name and 'Rastplatz' not in junction_name and 'Parkplatz' not in junction_name:
            motorway_junction_info_for_highway.append({'highway': name, 'name': junction_name,
                                                   'osm_id': junctions.osm_id.to_list()[ij],
                                                   'geometry': junction_geom.wkt, 'asfinag_position': junctions.ref.to_list()[ij],
                                                   'dist_along_highway': highway_geom.project(junction_geom)})

    motorway_junction_info_for_highway = pd.DataFrame(motorway_junction_info_for_highway)
    if len(motorway_junction_info_for_highway) > 0:
        motorway_junction_info_for_highway = motorway_junction_info_for_highway.sort_values(by=['dist_along_highway'])
        motorway_junction_info_for_highway = motorway_junction_info_for_highway.drop_duplicates(subset=['name', 'asfinag_position'])
        motorway_junction_info = motorway_junction_info.append(motorway_junction_info_for_highway)


motorway_junction_info.to_csv('data/motorway_junctions_a9.csv', index=False)


