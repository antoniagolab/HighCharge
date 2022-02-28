"""

demand calculation


"""
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
from _utils import *
from _geometry_utils import *
import time
from _file_import_demcalc import *
# supress warnings
pd.options.mode.chained_assignment = None  # default='warn'

# def calc_demand(timestamp, sp_dem):
timestamp = time.strftime("%Y%m%d-%H%M%S")

# collecting traffic counts
tc_output = retrieve_tc_data_for_all_segments_for_dir(
    segments_gdf, tc_gdf, asfinag_data, vehicle_type="Kfz <= 3,5t hzG"
)
tc_gdf["tc_dir_0"] = tc_output.dir_0.to_list()
tc_gdf["tc_dir_1"] = tc_output.dir_1.to_list()


# producing pois_df for all segments
# poi (=point of interest) encompass all points along the segments at which traffic counts estimates are to be estimated
# for further demand estimation

pois_df = pd.DataFrame()
for seg_id in segment_ids:
    pois_for_seg = retrieve_demand_pois_for_segm(
        seg_id, segments_gdf, links_gdf, rest_areas
    )
    pois_df = pois_df.append(pois_for_seg)

pois_df["ID"] = range(0, len(pois_df))
pois_df = pois_df.set_index("ID")
# calculate demands

pois_df = pois_df.sort_values(by=["segment_id", "dist_along_segment"])

(
    pois_df["demand_0"],
    pois_df["demand_1"],
    pois_df["tc_0"],
    pois_df["tc_1"],
) = calculate_demand(pois_df, tc_gdf, col_tc_dir_0="tc_dir_0", col_tc_dir_1="tc_dir_1")

# pois_df.to_csv("data/" + timestamp + "_demand_calculated.csv")
pois_df["ID"] = range(0, len(pois_df))
pois_df = pois_df.set_index("ID")
pois_df.to_csv("data/_demand_calculated_2.csv")


# ----------
# generating sequence of node visits

dist_max = segments_gdf.length.sum() / 3

# direction = 0

path_dictionaries = {}

seg_ids = segments_gdf.ID.to_list()
for id in seg_ids:
    dir = 0
    name = str(id) + "_" + str(dir) + "_0"
    segm_tree = {name: Node((id, dir))}
    unfiltered_path = get_children_from_seg(
        id,
        name,
        dir,
        segments_gdf,
        links_gdf,
        pois_df,
        segm_tree,
        0,
        dist_max,
        stop_then=False,
    )

    path = filter_path(unfiltered_path, segments_gdf, name)
    path_dictionaries[str(id) + "_" + str(dir)] = path
    print(id, dir, "done")

a_file = open("data/path_data.pkl", "wb")
pickle.dump(path_dictionaries, a_file)
a_file.close()

# a big dictionary with a dict for each seg_id and direction

path_dictionaries = {}

seg_ids = segments_gdf.ID.to_list()
for id in seg_ids:
    dir = 1
    name = str(id) + "_" + str(dir) + "_0"
    segm_tree = {name: Node((id, dir))}
    unfiltered_path = get_children_from_seg(
        id,
        name,
        dir,
        segments_gdf,
        links_gdf,
        pois_df,
        segm_tree,
        0,
        dist_max,
        stop_then=False,
    )

    path = filter_path(unfiltered_path, segments_gdf, name)
    path_dictionaries[str(id) + "_" + str(dir)] = path
    print(id, dir, "done")

a_file = open("data/path_data_2.pkl", "wb")
pickle.dump(path_dictionaries, a_file)
a_file.close()

