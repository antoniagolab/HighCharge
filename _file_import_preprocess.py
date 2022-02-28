import numpy as np
import pandas as pd
import geopandas as gpd
from _utils import clean_segments, filter_segments, split_by_dir, pd2gpd, edit_asfinag_file
from _variable_definitions import *
import pickle

# ---------------------------------------------------------------------------------------------------------------------
# Data for pre-processing
# ---------------------------------------------------------------------------------------------------------------------

# This file specifies of which highway (by name) which directed highway line is to be taken in order to obtain a
# continuous representation of the street network by one one line

orientation_info = pd.read_csv(
    "C:/Users\golab\PycharmProjects/trafficflow\data/highway orientations.csv"
)

# shape with singular highways and motorways
highway_geometries = gpd.read_file("geometries/highway_geometries_v9.shp")
# merged highway network to one shape
merged_ = gpd.read_file("geometries/merg_v12.shp")


# rest areas with geometries represented by centroid for two driving directions as specified by ASFINAG
rest_areas_0 = pd2gpd(
    pd.read_csv("data/rest_areas_with_centroids_0.csv"), geom_col_name="centroid"
)
rest_areas_1 = pd2gpd(
    pd.read_csv("data/rest_areas_with_centroids_1.csv"), geom_col_name="centroid"
)

# geometric information on traffic counters along high-level road network
tcs = gpd.read_file("geometries/traffic_counters_positions_v26.shp")


