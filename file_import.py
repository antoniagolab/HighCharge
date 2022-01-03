import pandas as pd
import geopandas as gpd
from utils import clean_segments, filter_segments, split_by_dir, pd2gpd
from variable_definitions import *

pois_df = pd.read_csv(
    "data/_demand_calculated.csv"
)  # file containing POIs (points of interest) which
# highway crossing and service stations (no parking spots at
# this point
pois_df["POI_ID"] = range(0, len(pois_df))
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))
links_gdf = pd2gpd(pd.read_csv("data/highway_intersections.csv"))
links_gdf = links_gdf[~(links_gdf.index == 19)]
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

