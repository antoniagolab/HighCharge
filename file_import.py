import numpy as np
import pandas as pd
import geopandas as gpd
from utils import clean_segments, filter_segments, split_by_dir, pd2gpd
from variable_definitions import *
import pickle

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
#     [0, 1, 2])
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

n0 = len(dir_0)
n1 = len(dir_1)
k = len(dir)
n3 = k


ex_infr_0 = pd.read_csv("data/rest_areas_0_charging_stations_old.csv")
ex_infr_1 = pd.read_csv("data/rest_areas_1_charging_stations_old.csv")


# join this with rest areas
rest_areas = pd2gpd(
    pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
).sort_values(by=["on_segment", "dist_along_highway"])
rest_areas["segment_id"] = rest_areas["on_segment"]
rest_areas[col_type_ID] = rest_areas["nb"]
rest_areas[col_directions] = rest_areas["evaluated_dir"]

existing_infr = pd.merge(rest_areas, ex_infr_0, on=[col_highway, 'name', 'direction'])
# rest_areas_0, rest_areas_1 = split_by_dir(rest_areas, col_directions, reindex=True)

ex_infr = ex_infr_0.append(ex_infr_1)
ex_infr = ex_infr.drop_duplicates(subset=['highway', 'name', 'direction'])
ex_infr = ex_infr.sort_values(by=['highway', 'name'])
#
# existing_infr_0 = pd.merge(rest_areas_0, ex_infr_0, on=[col_highway, 'name', 'direction'])
existing_infr = pd.merge(rest_areas, ex_infr, on=[col_highway, 'name', 'direction'])

# existing_infr = existing_infr_0.append(existing_infr_1)
existing_infr = existing_infr.drop_duplicates(subset=['highway', 'name', 'asfinag_position'])
existing_infr = existing_infr.sort_values(by=['highway', 'name'])
existing_infr = existing_infr[existing_infr.has_charging_station == True]
existing_infr = existing_infr.replace(np.NAN, 0)
# existing_infr['installed_infrastructure'] =existing_infr['350kW']
existing_infr_0, existing_infr_1 = split_by_dir(existing_infr, 'dir', reindex=True)

installed_cap = existing_infr['50kW'] * 50 + existing_infr['75kW'] * 75 + existing_infr['150kW'] * 150 + existing_infr['350kW'] * 350

