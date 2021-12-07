"""

Definition of demand-related parameters

"""
import pandas as pd
from utils import *

from edit_asfinag_data import *


# get data

segments_gdf = pd2gpd(pd.read_csv('data/highway_segments.csv'))
links_gdf = pd2gpd(pd.read_csv('data/highway_intersections.csv'))
tc_gdf = pd.read_csv('data/tcs_v1.csv')
tc_dir_info = pd.read_csv('data/tcs_v1_edited.csv')
tc_gdf = pd2gpd(pd.merge(tc_gdf, tc_dir_info[['nb', 'Dir==0']], on=['nb'])).sort_values(by=['on_segment',
                                                                                                     'dist_along_segm'])

rest_areas = pd2gpd(pd.read_csv("data/projected_ras.csv"), geom_col_name='centroid').sort_values(by=['on_segment',
                                                                                                     'dist_along_highway'])
segment_ids = list(set(segments_gdf.ID.to_list()))


# input traffic count numbers
input_tc_file = "data/Verkehrsstatistik Juni 2021"
edit_asfinag_file(input_tc_file)
asfinag_data = pd.read_csv(input_tc_file + "_edited.csv")

#
spec_demand = 0.2   # (kWh/100km)


# column definitions
col_direction = "evaluated_dir"

