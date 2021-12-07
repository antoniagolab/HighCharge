import geopandas as gpd
import pandas as pd
from shapely import wkt

reference_coord_sys = "EPSG:31287"

# import edited traffic counter positions along the highway
tc = pd.read_csv('data/traffic_counters_positions_v22.csv')
tc_new = tc

# get highway geometries
highway_geometries = pd.read_csv(r'geometries/highway_geometries_v6.csv')
highway_geometries['geometry'] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)

highway_names_of_tcs = tc.highway.to_list()
edited_dists = tc.position.to_list()
inds = tc.index.to_list()

for ij in range(0, len(tc)):
    highway = highway_names_of_tcs[ij]
    if highway == 'A9':
        extract_hg = highway_geometries[highway_geometries.highway == highway]
        highway_geom = extract_hg.geometry.to_list()[0]
        dist = edited_dists[ij]
        geom = highway_geom.interpolate(dist)
        current_ind = inds[ij]
        tc_new.at[current_ind, 'geometry'] = geom.wkt

tc_new.to_csv('data/traffic_counters_positions_v24.csv')
tc_new['geometry'] = tc_new.geometry.apply(wkt.loads)
tc_new = gpd.GeoDataFrame(tc_new, crs=reference_coord_sys, geometry='geometry')
tc_new.to_file('geometries/traffic_counters_positions_v24.shp')




