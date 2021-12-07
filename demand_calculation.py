"""

demand calculation


"""
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
from optimization_parameters import *
from demand_related_parameters import *
from utils import *
from geometry_utils import *
import time

# supress warnings
pd.options.mode.chained_assignment = None  # default='warn'

# def calc_demand(timestamp, sp_dem):
timestamp = time.strftime("%Y%m%d-%H%M%S")
# TODO:
#   -   get all relevant data and sort it after segment and distance along segment  -check
#   -   get traffic counts  -check
#   -   produce pois df -check
#   -   calculate demand at these pois  -check
#   -   save with timestamp and forward -check
#   -   create dir_0 and dir_1  -check
#   -   create mask and enum (for 0 and 1) - produce a "neighbour" matrix first; where 1 for each neighbouring
#   -   I need the following matrices: (1) ENUM_0, ENUM_1 + sim. create with ENUM_dirs_0, ENUM_dirs_1 which indicate
#      the driving direction at the ra, (2) DIST_0, DIST_1 indicating the distances of (ij,ij) to all other; (3) MASK_0,
#       MASK_1 created based on the d_max and DIST_0, DIST_1
#       -BE CAREFUL: how to deal with links? they have more than two directions? probably do not introduce them as nodes
#       to optimization -> just in dividing the demand
#       what has to be done: input to first ra after link: total local demand= local demand + before links
#       no, actually it is better to include the links and then just being careful with charged and demand relationship
#   - overthink issue of circulating energy demand!! Is it solved by simply removing already visited segments in this branch
#   - is demand getting lost?!!??!! -> but doesn't it make sense, that the demand won't loop? -> assuming travel will
#   never continue to visit the same segment again; but then: I need to be careful about when a certain segment is visited!
#   - from which direction; and maybe I have to recalculate this path for each direction and segment
#   - I can use Walker to create input-output relationships
#   - save traffic counts + demands for both directions -check
#   - introducing also links as nodes but setting X=0 for these; also charged = 0; but just for nodes of splits
#   - or not? because there is some not-covered energy; but could still output = 0
#   - Be careful: links are double or multiple times inside => equal these later
#   - GRNN not nice!

# collecting traffic counts
tc_output = retrieve_tc_data_for_all_segments_for_dir(segments_gdf, tc_gdf, asfinag_data, vehicle_type="Kfz <= 3,5t hzG")
tc_gdf['tc_dir_0'] = tc_output.dir_0.to_list()
tc_gdf['tc_dir_1'] = tc_output.dir_1.to_list()


# producing pois_df for all segments
# poi (=point of interest) encompass all points along the segments at which traffic counts estimates are to be estimated
# for further demand estimation

pois_df = pd.DataFrame()
for seg_id in segment_ids:
    pois_for_seg = retrieve_demand_pois_for_segm(seg_id, segments_gdf, links_gdf, rest_areas)
    pois_df = pois_df.append(pois_for_seg)

pois_df['ID'] = range(0, len(pois_df))
pois_df = pois_df.set_index('ID')
# calculate demands

pois_df = pois_df.sort_values(by=['segment_id', 'dist_along_segment'])

pois_df["demand_0"], pois_df["demand_1"], pois_df["tc_0"], pois_df["tc_1"] = calculate_demand(pois_df, tc_gdf, sp_demand=spec_demand, col_tc_dir_0='tc_dir_0', col_tc_dir_1='tc_dir_1')

# pois_df.to_csv("data/" + timestamp + "_demand_calculated.csv")
pois_df['ID'] = range(0, len(pois_df))
pois_df = pois_df.set_index('ID')
pois_df.to_csv("data/_demand_calculated.csv")



