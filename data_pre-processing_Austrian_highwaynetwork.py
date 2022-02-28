"""
This file serves to prepare data on highway network geometries and rest area geometries for optimization


last change: 28.02.2022

"""


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import (
    MultiLineString,
    LineString,
    LinearRing,
    MultiPoint,
    GeometryCollection,
)
from shapely.ops import split, linemerge
# from edit_asfinag_data import *
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
from _geometry_utils import *
from _utils import *
import ast
from _file_import_preprocess import *

# read data

one_network_gdf = gpd.GeoDataFrame()

highway_names = orientation_info.highway.to_list()
normal = orientation_info.normal.to_list()

for ij in range(0, len(orientation_info)):
    n = normal[ij]
    h = highway_names[ij]
    extract = highway_geometries[
        (highway_geometries.highway == h) & (highway_geometries.normal == n)
    ]
    one_network_gdf = one_network_gdf.append(extract)

# saving this to a file
one_network_gdf.to_file("geometries/one_network.shp")


# making this into a network to identify intersections

geoms = one_network_gdf.geometry.to_list()
one_network_geometry = MultiLineString(geoms)


# find intersections through intersecting all segments of one_network_gdf

highways = one_network_gdf.geometry.to_list()
n = len(highways)

intersection_geometries = []

for ij in range(0, n):
    for kl in range(0, n):
        if kl > ij:
            if highways[ij].intersects(highways[kl]):
                intersection_point = highways[ij].intersection(highways[kl])
                intersection_geometries.append(intersection_point)


# split, then delete small segments, then save these

# rounding all

collection_network = wkt.loads(
    wkt.dumps(GeometryCollection(one_network_geometry), rounding_precision=4)
)
collection_intersection = wkt.loads(
    wkt.dumps(MultiPoint(intersection_geometries), rounding_precision=4)
)
lm = linemerge(MultiLineString(collection_network))
g = merged_.geometry.to_list()[0]
splitted = split(g, MultiPoint(collection_intersection))
filtered_splits = [s for s in splitted if s.length > 50]
# round again
collection_filtered_splits = wkt.loads(
    wkt.dumps(GeometryCollection(filtered_splits), rounding_precision=4)
)
filtered_splits = [geom for geom in collection_filtered_splits]

# sort segments from left to right

sorted_left_right_segments, _ = sort_left_to_right(filtered_splits)

# sort points of segments from left to right
segments_l2r = [sort_points_in_linestring_l2r(l) for l in sorted_left_right_segments]

# creating GeoDataFrame with segments
segments_gdf = gpd.GeoDataFrame(
    {
        "ID": range(0, len(sorted_left_right_segments)),
        "geometry": segments_l2r,
    },
    crs=reference_coord_sys,
)

# sorting intersections left to right
sorted_left_right_inters, _ = sort_left_to_right(intersection_geometries)
del sorted_left_right_inters[26]
del sorted_left_right_inters[16]
# del sorted_left_right_inters[17]

sorted_left_right_inters.append(Point(611103.7492, 502111.9365))
sorted_left_right_inters, _ = sort_left_to_right(sorted_left_right_inters)

# finding connecting segments
connections = finding_segments_point_lies_on(sorted_left_right_inters, segments_gdf)
connecting_edges = list(connections.values())
number_of_edges = [len(l) for l in connecting_edges]

# creating GeoDataFrame with all intersections
intersections_gdf = gpd.GeoDataFrame(
    {
        "ID": range(0, len(sorted_left_right_inters)),
        "geometry": sorted_left_right_inters,
        "nb_of_conn_edges": number_of_edges,
        "conn_edges": connecting_edges,
    },
    crs=reference_coord_sys,
)

# now define link_0 and link_1

for id in segments_gdf.ID.to_list():
    link_0 = None
    link_1 = None
    for kl in intersections_gdf.ID.to_list():
        if id in connecting_edges[kl]:
            if (
                np.absolute(
                    get_leftmost_end_point(segments_l2r[id])
                    - sorted_left_right_inters[kl].x
                )
                < 1e-3
            ):
                link_0 = kl
            elif (
                np.absolute(
                    get_rightmost_end_point(segments_l2r[id])
                    - sorted_left_right_inters[kl].x
                )
                < 1e-3
            ):
                link_1 = kl
    segments_gdf.at[id, "link_0"] = link_0
    segments_gdf.at[id, "link_1"] = link_1

# save intermediate states
intersections_gdf.to_csv("data/highway_intersections.csv", index=False)
segments_gdf.to_csv("data/segments.csv", index=False)

intersections_gdf[["ID", "geometry"]].to_file("data/highway_intersections_v1.shp")
segments_gdf.to_file("geometries/segments_v1.shp")

# TODO: identifying closest traffic counters to each link for each edge
#   - get traffic counter positions -check
#   - snap them to network -check
#   - (1) from each intersection, get meeting edges/segments, (2) determine whether these are link_0 or link_1
#   - because when "==link_1 " is True, then search for last tc, otherwise first.
#   - find closest ones to each intersection

segments_gdf = pd2gpd(pd.read_csv("data/segments.csv"))
intersections_gdf = pd2gpd(pd.read_csv("data/highway_intersections.csv"))

rest_areas_0 = rest_areas_0[~(rest_areas_0.asfinag_type == 2)]
rest_areas_1 = rest_areas_1[~(rest_areas_1.asfinag_type == 2)]

segment_geometries = segments_gdf.geometry.to_list()

# project rest areas
rest_areas = unite_dfs(
    rest_areas_0, rest_areas_1, drop_duplicate_list=["name", "highway", "direction"]
)
rest_areas = rest_areas.sort_values(by="asfinag_position")
rest_areas["ID"] = range(0, len(rest_areas))  # indexing the rest areas
rest_areas = rest_areas.set_index("ID")  # indexing the rest areas

# sorting l2r
rest_areas["x_coord"] = [p.x for p in rest_areas.geometry.to_list()]
rest_areas = rest_areas.sort_values(by="x_coord")
rest_areas["ID"] = range(0, len(rest_areas))  # indexing the rest areas
rest_areas = rest_areas.set_index("ID")  # indexing the rest areas
rest_areas["nb"] = range(0, len(rest_areas))


projected_ras = rest_areas.copy()
projected_ras["centroid"] = projected_ras.apply(
    lambda row: MultiLineString(segments_gdf.geometry.to_list()).interpolate(
        MultiLineString(segments_gdf.geometry.to_list()).project(row.centroid)
    ),
    axis=1,
)

# finding belonging segment
points_on_lines = finding_segments_point_lies_on(
    projected_ras.geometry.to_list(), segments_gdf
)
line_ids = list(points_on_lines.values())
tc_on_line_ids = [p[0] for p in line_ids]
projected_ras["on_segment"] = tc_on_line_ids
rest_areas["on_segment"] = tc_on_line_ids


# determining of direction:
# TODO: for each ra -> extract belonging segment -> project onto and set 10 m further another point
#   make classification of left/right
ra_geometries = rest_areas.geometry.to_list()
ra_segment_ids = rest_areas.on_segment.to_list()
ra_directions = rest_areas.direction.to_list()

dir_info = []
for ij in range(0, len(rest_areas)):
    if not ra_directions[ij] == 2:
        direc = classify_dir(segment_geometries[ra_segment_ids[ij]], ra_geometries[ij])
    else:  # rest area is for both directions
        direc = 2
    dir_info.append(direc)

rest_areas["evaluated_dir"] = dir_info
projected_ras["evaluated_dir"] = dir_info

# calculate distance along the route
rest_areas["dist_along_segm"] = calculate_distances_along_segments(
    segments_gdf, rest_areas
)

# traffic counters
tcs["x_coord"] = [p.x for p in tcs.geometry.to_list()]
tcs = tcs.sort_values(by="x_coord")
tcs["ID"] = range(0, len(tcs))
tcs = tcs.set_index("ID")


# projecting these onto network
projected_tcs = tcs.copy()
projected_tcs["geometry"] = projected_tcs.apply(
    lambda row: MultiLineString(segment_geometries).interpolate(
        MultiLineString(segment_geometries).project(row.geometry)
    ),
    axis=1,
)

# determine segments on which the tcs where projected
points_on_lines = finding_segments_point_lies_on(
    projected_tcs.geometry.to_list(), segments_gdf
)
line_ids = list(points_on_lines.values())
tc_on_line_ids = [p[0] for p in line_ids]
projected_tcs["on_segment"] = tc_on_line_ids
projected_tcs.to_csv("data/projected_tcs_v0.csv", index=False)
connecting_edges = [ast.literal_eval(s) for s in intersections_gdf.conn_edges.to_list()]


links_0 = segments_gdf.link_0.to_list()
links_1 = segments_gdf.link_1.to_list()


nearest_tcs_dic = {}
segm_has_tc_dic = {}
for ij in range(0, len(intersections_gdf)):
    # get edge ids
    # determine whether link_0/1
    nearest_tcs = []
    conn_edges = connecting_edges[ij]
    for c in conn_edges:
        current_segment = segment_geometries[c]
        tcs_on_segm = projected_tcs[projected_tcs["on_segment"] == c]
        tc_nbs = tcs_on_segm.nb.to_list()
        tcs_on_segm["dist"] = tcs_on_segm.geometry.apply(current_segment.project)
        if len(tcs_on_segm) > 0:
            if ij == links_0[c]:
                idx = tcs_on_segm["dist"].argmin()
                nearest_tcs.append(tc_nbs[idx])
            elif ij == links_1[c]:
                idx = tcs_on_segm["dist"].argmax()
                nearest_tcs.append(tc_nbs[idx])
            segm_has_tc_dic[c] = True
        else:
            nearest_tcs.append(np.nan)
    nearest_tcs_dic[ij] = nearest_tcs

intersections_gdf["neighbour_tcs"] = list(nearest_tcs_dic.values())

has_tc = []
for ij in range(0, len(segments_gdf)):
    if ij in list(segm_has_tc_dic.keys()):
        has_tc.append(True)
    else:
        has_tc.append(False)

segments_gdf["has_tc"] = has_tc

# evaluating nearest rest areas to intersections

nearest_ras_dic = {}
segm_has_ra_dic = {}
for ij in range(0, len(intersections_gdf)):
    # get edge ids
    # determine whether link_0/1
    nearest_ras = []
    conn_edges = connecting_edges[ij]
    for c in conn_edges:
        current_segment = segment_geometries[c]
        ra_on_segm = projected_ras[projected_ras["on_segment"] == c]
        ra_nbs = ra_on_segm.nb.to_list()
        ra_on_segm["dist"] = ra_on_segm.geometry.apply(current_segment.project)
        if len(ra_on_segm) > 0:
            if ij == links_0[c]:
                idx = ra_on_segm["dist"].argmin()
                nearest_ras.append(ra_nbs[idx])
            elif ij == links_1[c]:
                idx = ra_on_segm["dist"].argmax()
                nearest_ras.append(ra_nbs[idx])
            segm_has_ra_dic[c] = True
        else:
            nearest_ras.append(np.nan)
    nearest_ras_dic[ij] = nearest_ras

intersections_gdf["neighbour_ras"] = list(nearest_ras_dic.values())


has_ra = []
for ij in range(0, len(segments_gdf)):
    if ij in list(segm_has_ra_dic.keys()):
        has_ra.append(True)
    else:
        has_ra.append(False)

segments_gdf["has_ra"] = has_ra

segments_gdf.to_csv("data/highway_segments.csv", index=False)
intersections_gdf.to_csv("data/highway_intersections.csv", index=False)
projected_ras.to_csv("data/projected_ras.csv", index=False)
projected_tcs["dist_along_segm"] = calculate_distances_along_segments(
    segments_gdf, projected_tcs
)
projected_tcs.to_csv("data/tcs_v1.csv", index=False)

