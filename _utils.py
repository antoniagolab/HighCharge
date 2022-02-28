import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from _geometry_utils import *
from neupy import algorithms
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, findall, search
import ast
from os import listdir
from os.path import isfile, join


def pd2gpd(dataframe, geom_col_name="geometry"):
    dataframe[geom_col_name] = dataframe[geom_col_name].apply(wkt.loads)
    dataframe = gpd.GeoDataFrame(dataframe, geometry=geom_col_name)
    return dataframe.set_crs(reference_coord_sys)


def unite_dfs(dataframe1, dataframe2, drop_duplicate_list=[]):
    df = dataframe1.append(dataframe2).drop_duplicates(subset=drop_duplicate_list)
    df["ID"] = range(0, len(df))
    df = df.set_index("ID")
    return df


def retrieve_tc_data_for_all_segments_for_dir(
    segments_gdf,
    traffic_counter_df,
    asfinag_data,
    vehicle_type="Kfz <= 3,5t hzG",
    tc_column="Kfz/24h",
):
    """

    :param segments_gdf: gpd.GeoDataFrame with all segment geometries
    :param traffic_counter_df: pandas.DataFrame ("nb": traffic counter identifier by ASFINAG, "on_segment": belonging
        segment_id referring to segments in segment_gdf)
    :param asfinag_data: pandas.Dataframe, edited with edit_asfinag_data.edit_asfinal_file()
    :param vehicle_type: String, "Kfz <= 3,5t hzG" or "Kfz > 3,5t hzG"
    :param tc_column: column name containing traffic count numbers in asfinag_data
    :return: pandas.DataFrame ("dir_0": extracted tc numbers matching to traffic_counter_df in direction 0,
        "dir_1": extracted tc numbers matching to traffic_counter_df in direction 0)
    """
    segment_ids = list(set(segments_gdf.ID.to_list()))
    for seg_id in segment_ids:
        filtered_asfinag_data = asfinag_data[asfinag_data["Unnamed: 6"] == vehicle_type]
        tcs_on_segment = traffic_counter_df[traffic_counter_df["on_segment"] == seg_id]
        segment_geom = segments_gdf[segments_gdf.ID == seg_id].geometry.to_list()[0]

        if not len(tcs_on_segment) == 0:

            indices = tcs_on_segment.index.to_list()
            # retrieving distances along segment
            dists_along_tc = [
                segment_geom.project(tc) for tc in tcs_on_segment.geometry.to_list()
            ]
            tcs_on_segment["dist_along_segm"] = dists_along_tc
            dir0 = tcs_on_segment["Dir==0"].to_list()
            tcs_ids = tcs_on_segment.nb.to_list()

            # retrieving traffic counts
            traffic_count_numbers_dir_0 = []
            traffic_count_numbers_dir_1 = []
            for ij in range(0, len(tcs_ids)):
                id = tcs_ids[ij]
                extract_tc = filtered_asfinag_data[
                    filtered_asfinag_data["Unnamed: 3"] == id
                ]
                if len(extract_tc):
                    numbers = extract_tc[tc_column].to_list()
                    if dir0[ij] == 0:
                        traffic_count_numbers_dir_0.append(numbers[0])
                        traffic_count_numbers_dir_1.append(numbers[1])
                    else:
                        traffic_count_numbers_dir_0.append(numbers[1])
                        traffic_count_numbers_dir_1.append(numbers[0])
                else:
                    traffic_count_numbers_dir_0.append(np.NaN)
                    traffic_count_numbers_dir_1.append(np.NaN)

            traffic_counter_df.at[indices, "tc_dir_0"] = traffic_count_numbers_dir_0
            traffic_counter_df.at[indices, "tc_dir_1"] = traffic_count_numbers_dir_1

    return pd.DataFrame(
        {
            "dir_0": traffic_counter_df.tc_dir_0.to_list(),
            "dir_1": traffic_counter_df.tc_dir_1.to_list(),
        }
    )


def retrieve_demand_pois_for_segm(segment_id, segments_gdf, links_gdf, ra_gdf):
    """
    function to retrieve all distances along a segment on which the demand is of interest: These include link_0, link_1
    and all existing rest areas
    :param segment_id:
    :return: dataframe with distance information along the segment
    """
    link_ids = links_gdf.ID.to_list()
    segment_geom = segments_gdf[segments_gdf.ID == segment_id].geometry.to_list()[0]
    link_0_info = segments_gdf[segments_gdf.ID == segment_id].link_0.to_list()[0]
    link_1_info = segments_gdf[segments_gdf.ID == segment_id].link_1.to_list()[0]
    rest_areas_along_segm = ra_gdf[ra_gdf["on_segment"] == segment_id]

    # adding link 0 information
    pois_type = []
    ID = []
    dist = []
    direction = []
    pois_type.append("link")
    dist.append(0)
    direction.append(2)
    if link_0_info in link_ids:
        ID.append(link_0_info)
    else:
        ID.append(np.NaN)

    # adding rest area information if exists
    if len(rest_areas_along_segm) > 0:
        nbs = rest_areas_along_segm.nb.to_list()
        geoms = rest_areas_along_segm.geometry.to_list()
        dirs = rest_areas_along_segm.evaluated_dir.to_list()
        for kl in range(0, len(rest_areas_along_segm)):
            pois_type.append("ra")
            ID.append(nbs[kl])
            dist.append(segment_geom.project(geoms[kl]))
            direction.append(dirs[kl])

    # adding link 1 information
    pois_type.append("link")
    dist.append(segment_geom.length)
    direction.append(2)
    if link_1_info in link_ids:
        ID.append(link_1_info)
    else:
        ID.append(np.NaN)

    return pd.DataFrame(
        {
            "segment_id": [segment_id] * len(ID),
            "pois_type": pois_type,
            "type_ID": ID,
            "dir": direction,
            "dist_along_segment": dist,
        }
    )


def calculate_integral(neural_network, lb, ub, res=0.05):
    """

    :param neural_network: algorithm.GRNN object
    :param lb: x value of lower bound of integral
    :param ub: x value of upper bound of integral
    :param res: resolution of integral calculation(default: 0.1)
    :return:
    """
    x_values = np.arange(lb, ub, res)
    y_pred = neural_network.predict(x_values)
    y_values = [l[0] for l in y_pred]
    integral = np.trapz(y_values, dx=res)
    return integral


def calculate_demand(pois_df, tc_df, col_tc_dir_0, col_tc_dir_1, std=5000):
    """
    calculating demand for each poi along a segment
    :param pois_df: pandas.DataFrame; pois with their distances along a segment
    :param tc_df: pandas.DataFrame
    :param col_tc_dir_0: column in tc_df indicating the traffic counts
    :param col_tc_dir_1: column in tc_df indicating the traffic counts
    :param std
    :return: column "energy_demand" in length pois_df
    """
    col_distance = "dist_along_segment"
    # clear missing data    -check
    # create network
    # for each direction: calculate demand at each pois (integral * sp_demand)
    #
    pois_df = pois_df.sort_values(by=["segment_id", col_distance])
    segment_ids = list(set(pois_df.segment_id.to_list()))

    # clear data
    cleaned_tc_df = tc_df[tc_df[col_tc_dir_0] > 0]
    for seg_id in segment_ids:
        extract_tc = cleaned_tc_df[cleaned_tc_df.on_segment == seg_id]
        extract_tc = extract_tc.sort_values(by="dist_along_segm")

        if len(extract_tc) > 0:
            # create neural network
            x_values = np.array(extract_tc["dist_along_segm"].to_list())
            y_values_0 = np.array(extract_tc[col_tc_dir_0].to_list())
            y_values_1 = np.array(extract_tc[col_tc_dir_1].to_list())

            fun0 = algorithms.GRNN(std=std)
            fun0.train(x_values, y_values_0)

            fun1 = algorithms.GRNN(std=std)
            fun1.train(x_values, y_values_1)

            poi_extract = pois_df[pois_df.segment_id == seg_id]

            poi_extract_dir_0, poi_extract_dir_1 = split_by_dir(
                poi_extract, "dir", reindex=False
            )

            inds_0 = poi_extract_dir_0.index.to_list()
            inds_1 = poi_extract_dir_1.index.to_list()

            dists_0 = poi_extract_dir_0[col_distance].to_list()
            dists_1 = poi_extract_dir_1[col_distance].to_list()

            lb_x = 0
            demand_dir_0 = [0]

            # direction 0
            for kl in range(1, len(poi_extract_dir_0)):
                ub_x = dists_0[kl]
                integral = calculate_integral(fun0, lb_x, ub_x)
                demand_dir_0.append(integral * (1 / 100000))
                lb_x = ub_x

            ub_x = dists_1[-1]
            demand_dir_1 = []
            # direction 1
            for kl in range(len(poi_extract_dir_1) - 2, -1, -1):
                lb_x = dists_1[kl]
                integral = calculate_integral(fun1, lb_x, ub_x)
                demand_dir_1.append(integral * (1 / 100000))
                ub_x = lb_x
            demand_dir_1.append(0)

            traffic_counts_0 = fun0.predict(dists_0)
            traffic_counts_1 = fun1.predict(dists_1)
            plot_tc_GRNN(seg_id, 0, x_values, y_values_0, dists_0, traffic_counts_0)
            plot_tc_GRNN(seg_id, 1, x_values, y_values_1, dists_1, traffic_counts_1)

            pois_df.at[inds_0, "demand_0"] = demand_dir_0
            pois_df.at[inds_1, "demand_1"] = demand_dir_1
            pois_df.at[inds_0, "tc_0"] = traffic_counts_0
            pois_df.at[inds_1, "tc_1"] = traffic_counts_1
            print(seg_id, " done")

    return (
        pois_df.demand_0.to_list(),
        pois_df.demand_1.to_list(),
        pois_df.tc_0.to_list(),
        pois_df.tc_1.to_list(),
    )


def split_by_dir(df, col_dir, reindex=True):
    """
    function splitting a dataframe by direction into two dataframes
    :param df:  pandas.DataFrame
    :param col_dir: String of column name in which driving direction is specified (0=direction 0, 1= dir. 1, 2 = dir 0
        + dir 1)
    :return: two pandas.DataFrame
    """
    dir_0 = df[df[col_dir].isin([0, 2])]
    dir_1 = df[df[col_dir].isin([1, 2])]
    if reindex:
        dir_0["ind"] = range(0, len(dir_0))
        dir_1["ind"] = range(0, len(dir_1))
        dir_0 = dir_0.set_index("ind")
        dir_1 = dir_1.set_index("ind")

    return dir_0, dir_1


def plot_tc_GRNN(segment_id, dir, x_0, y_0, x_1, y_1):
    """

    :param segment_id:
    :param dir:
    :param x_0:
    :param y_0:
    :param x_1:
    :param y_1:
    :return:
    """
    plt.figure()
    plt.plot(x_0, y_0, "*-", label="traffic count by ASFINAG")
    plt.plot(x_1, y_1, "*-", label="estimated traffic counts at rest areas")
    plt.xlabel("distance (m)")
    plt.ylabel("traffic count")
    plt.title(str(segment_id) + " dir=" + str(dir))
    plt.legend()
    plt.savefig("traffic_counts/" + str(segment_id) + "_dir_" + str(dir) + ".png")
    plt.close()


def make_mask(
    pois_gdf,
    link_gdf,
):
    """

    :param pois_gdf:
    :param link_gdf:
    :return:
    """
    mask = np.array([])
    enum = np.array([])
    return mask, enum


def get_direction_from_poi(id, pois_gdf):
    seg_id = pois_gdf.loc[id].segment_id
    extract_pois = pois_gdf[pois_gdf.segment_id == seg_id]
    inds = extract_pois.index.to_list()

    if id == inds[0]:
        direction = 0
    else:
        direction = 1
    return direction, seg_id


def get_children_from_seg(
    seg_id,
    prev_name,
    direc,
    segments_gdf,
    links_gdf,
    pois_df,
    seg_tree,
    current_dist,
    dmax,
    stop_then=False,
):
    """
     TODO:
     do not visit a segment again in same direction!!
    :param seg_id: segment id
    :param prev_name: name of previously created Node instance
    :param direc: current travel direction of segment with segment id = seg_id; 0/1
    :param segments_gdf:
    :param links_gdf:
    :param pois_df:
    :param seg_tree: dictionary with all nodes
    :param current_dist: current distance travelled from last node on initial segment
    :param dmax: maximum travel distance
    :param stop_then: boolean; True if dmax is reached
    :return:
    """
    ancestor_segms = [n.name for n in seg_tree[prev_name].ancestors]
    # ancestor_segms = []
    if direc == 0:
        link_id = segments_gdf[segments_gdf.ID == seg_id].link_1.to_list()[0]
        if link_id >= 0:
            connected_segs = links_gdf[links_gdf.ID == link_id].conn_edges.to_list()[0]
            connected_segs = ast.literal_eval(connected_segs)
            connected_segs.remove(seg_id)
            connected_segs_touples = []

            for c in connected_segs:

                if link_id == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]:
                    n = (c, 0)
                    if not n in ancestor_segms:
                        connected_segs_touples.append((c, 0))

                elif link_id == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]:
                    n = (c, 1)
                    if not n in ancestor_segms:
                        connected_segs_touples.append((c, 1))

            if len(connected_segs_touples) > 0:
                for cn in connected_segs_touples:
                    c = cn[0]
                    g = segments_gdf[segments_gdf.ID == c].geometry.to_list()[0]
                    current_dist = current_dist + g.length
                    if (
                        link_id
                        == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]
                    ):
                        name = str(c) + "_0_0"
                        ii = 0
                        while name in seg_tree.keys():
                            ii = ii + 1
                            name = str(c) + "_0_" + str(ii)

                        seg_tree[name] = Node((c, 0), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(
                                    c,
                                    name,
                                    0,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                    stop_then=True,
                                )
                            else:
                                get_children_from_seg(
                                    c,
                                    name,
                                    0,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                )

                    elif (
                        link_id
                        == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]
                    ):
                        name = str(c) + "_1_0"
                        ii = 0
                        while name in seg_tree.keys():
                            ii = ii + 1
                            name = str(c) + "_1_" + str(ii)
                        seg_tree[name] = Node((c, 1), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(
                                    c,
                                    name,
                                    1,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                    stop_then=True,
                                )
                            else:
                                get_children_from_seg(
                                    c,
                                    name,
                                    1,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                )

    else:
        link_id = segments_gdf[segments_gdf.ID == seg_id].link_0.to_list()[0]
        if link_id >= 0:
            connected_segs = links_gdf[links_gdf.ID == link_id].conn_edges.to_list()[0]
            connected_segs = ast.literal_eval(connected_segs)
            connected_segs.remove(seg_id)
            connected_segs_touples = []
            for c in connected_segs:
                if link_id == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]:
                    n = (c, 0)
                    if not n in ancestor_segms:
                        connected_segs_touples.append((c, 0))

                elif link_id == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]:
                    n = (c, 1)
                    if not n in ancestor_segms:
                        connected_segs_touples.append((c, 1))
            if len(connected_segs_touples) > 0:
                for cn in connected_segs_touples:
                    c = cn[0]
                    g = segments_gdf[segments_gdf.ID == c].geometry.to_list()[0]
                    current_dist = current_dist + g.length
                    if (
                        link_id
                        == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]
                    ):
                        name = str(c) + "_0_0"
                        ii = 0
                        while name in seg_tree.keys():
                            ii = ii + 1
                            name = str(c) + "_0_" + str(ii)
                        seg_tree[name] = Node((c, 0), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(
                                    c,
                                    name,
                                    0,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                    stop_then=True,
                                )
                            else:
                                get_children_from_seg(
                                    c,
                                    name,
                                    0,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                )
                    elif (
                        link_id
                        == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]
                    ):
                        name = str(c) + "_1_0"
                        ii = 0
                        while name in seg_tree.keys():
                            ii = ii + 1
                            name = str(c) + "_1_" + str(ii)
                        seg_tree[name] = Node((c, 1), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(
                                    c,
                                    name,
                                    1,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                    stop_then=True,
                                )
                            else:
                                get_children_from_seg(
                                    c,
                                    name,
                                    1,
                                    segments_gdf,
                                    links_gdf,
                                    pois_df,
                                    seg_tree,
                                    current_dist,
                                    dmax,
                                )
    return seg_tree


def filter_path(path, segments_gdf, origin_name):
    """
    filters the calculated path in order to ensure that a segment is not travelled a second time by the same demand
    this is done by ensuring to visit each segment only via the
    :param path: dictionary with all nodes
    :return: filtered path
    """
    path = path.copy()
    # print_tree(path[origin_name])
    name_list = []
    for k in path.keys():
        name_list.append(path[k].name[0])

    name_list = list(set(name_list))
    # print('name_list', name_list)
    shortest_lengths = []
    for name in name_list:
        keys_with_this_name = []
        for k in path.keys():
            if name == path[k].name[0] and path[origin_name] in path[k].ancestors:
                keys_with_this_name.append(k)
            if int(origin_name.split("_")[0]) == name:
                keys_with_this_name.append(k)
        lengths = []
        for k in keys_with_this_name:
            ancestors = path[k].ancestors
            seg_ids = []
            for a in ancestors:
                seg_id_dir = a.name[0]
                seg_ids.append(seg_id_dir)

            extract_segs = segments_gdf[segments_gdf.ID.isin(seg_ids)]
            lengths.append(extract_segs.length.sum())
        shortest_lengths.append(min(lengths))

    # sort names after shortest_lengths
    sorted_names = [x for _, x in sorted(zip(shortest_lengths, name_list))]
    # sorted_names = name_list
    for name in sorted_names:
        keys_with_this_name = []
        for k in path.keys():
            if name == path[k].name[0] and path[origin_name] in path[k].ancestors:
                keys_with_this_name.append(k)

        lengths = []
        for k in keys_with_this_name:
            ancestors = path[k].ancestors
            seg_ids = []
            for a in ancestors:
                seg_id_dir = a.name[0]
                seg_ids.append(seg_id_dir)

            extract_segs = segments_gdf[segments_gdf.ID.isin(seg_ids)]
            lengths.append(extract_segs.length.sum())

        sort_inds = np.argsort(lengths)
        to_del = sort_inds[1 : len(sort_inds)]
        for d in to_del:
            path[keys_with_this_name[d]].parent = None
            del path[keys_with_this_name[d]]
        clean_path_directory(path, origin_name)

    return path


def clean_path_directory(path, origin_name):
    path_descendants = path[origin_name].descendants
    keys = list(path.keys())
    for k in keys:
        if k in path.keys() and not k == origin_name:
            if not path[k] in path_descendants:
                del path[k]


def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))


def append_children_pois_on_same_segm(
    poi_id, segment_id, direction, pois_gdf, dist, last_name, counter, segm_tree, dmax
):
    dist_reached = False

    extract_pois = pois_gdf[pois_gdf.segment_id == segment_id]
    if direction == 1:
        extract_pois = extract_pois.sort_values(
            by="dist_along_segment", ascending=False
        )
        dists = extract_pois.dist_along_segment.to_list()
        extract_pois["dist_along_segment"] = [0] + [
            dists[-1] - d for d in dists[-1:0:-1]
        ]

    pois_indices = extract_pois.index.to_list()
    ind = pois_indices.index(poi_id)
    extract_pois = extract_pois.loc[ind : len(extract_pois)]
    pois_indices = extract_pois.index.to_list()[1 : len(extract_pois)]
    dist_init = extract_pois.dist_along_segment.to_list()
    dists = [
        d - dist_init
        for d in extract_pois.dist_along_segment.to_list()[1 : len(extract_pois)]
    ]

    cs = ["v_0"] + ["v_" + str(counter) + str(mn) for mn in range(1, len(extract_pois))]
    for kl in range(1, len(extract_pois)):
        dist = dist + dists[kl]
        if dist <= dmax:
            segm_tree[cs[kl]] = Node(
                (pois_indices[kl], direction, dist), parent=cs[kl - 1]
            )
        else:
            dist_reached = True
            break

    return segm_tree, dist, dist_reached, cs[-1], counter + len(extract_pois)


def append_children_pois(
    pois_gdf, segment_id, dir, dist, last_c, counter, variable_dic
):
    """

    :param pois_gdf:
    :param segment_id:
    :param dir:
    :param dist: distance of lastly inserted node at which to extend tree since origin node
    :param counter: counter of lastly inserted node at which to extend tree
    :param variable_dic:
    :return:
    """
    extract_seg_gdf = pois_gdf[pois_gdf.segment_id == segment_id]
    last_node = "v" + last_c
    # find this node
    if dir == 1:
        extract_seg_gdf = extract_seg_gdf.sort_values(
            by="dist_along_segment", ascending=False
        )
        dists = extract_seg_gdf.dist_along_segment.to_list()
        extract_seg_gdf["dist_along_segment"] = [0] + [
            dists[-1] - d for d in dists[-1:0:-1]
        ]

    distances = extract_seg_gdf.dist_along_segment.to_list()
    node_ids = extract_seg_gdf.index.to_list()

    chain = [last_node] + [
        "v" + str(mn + counter) for mn in range(0, len(extract_seg_gdf))
    ]
    for kl in range(1, len(chain)):
        dist = dist + distances[kl]
        variable_dic[chain[kl]] = Node((node_ids[kl], dir), parent=chain[kl - 1])

    return variable_dic, counter + len(chain) - 1


def append_for_last_link(
    io_rels, seg_id, direction, pois_df_0, pois_df_1, links_gdf, segments_gdf
):
    """
    this functions needs to be applied to enable to pass on the demand of the last link element on a highway segment to
     other segments

    :return:
    """
    # TODO:
    #   - check if needed (only needed if no IO-rels for last link on segment)
    #   - if needed -> check if this node is a linked to other segments
    #   - add ios to link the link to the starting pois of other segments
    #   - + add ios to link the link to first pois and ras next
    #   - actually I need to pass this energy on as l

    if direction == 0:
        pois_df = pois_df_0
    else:
        pois_df = pois_df_1

    extract_pois = pois_df[pois_df.segment_id == seg_id]

    # get last element
    last_ind = extract_pois.index.to_list[len(extract_pois)]

    filtered_ios = [io for io in io_rels if io[0] == (last_ind, direction)]
    new_ios = []
    if len(filtered_ios) > 0:
        return io_rels
    else:
        # what do I add?
        # just one step to next? -> if ra, then it makes sense; but if not, then this shift is to next link, then to
        # next link, ... -> resulting in issue of overlapping segments
        # for now: only to next node!

        if is_linkage(last_ind, direction, pois_df_0, pois_df_1)[0]:
            link_id = is_linkage(last_ind, direction, pois_df_0, pois_df_1)[1]
            equal_ids, _ = equal_linkages_points(pois_df_0, pois_df_1, links_gdf)
            equal_nodes = equal_ids[link_id]
            for e in equal_nodes:
                if not e == (last_ind, direction):
                    # adding io to equal node
                    new_ios.append(((last_ind, direction), ((last_ind, direction), e)))
                    # io needed to nearest node on the segment
                    # TODO: for this, I need to (1) get the segment id of the current node
                    #   (2) evaluate travel direction, (3) get poi extract based on segment id
                    #   (4) flip the extract if needed, (4) get second row of this data frame, (5) define io for this

        return None


def retrieve_demand_split_ratios(
    segment_id, direction, pois_df, links_gdf, segments_gdf, poi_dir_0, poi_dir_1
):
    """
    retrieves the ratios of splitting demand at a linkage
    :param segment_id:
    :param links_df:
    :param pois_df:
    :return:
    """
    tcs = []
    pois_extract = pois_df[pois_df.segment_id == segment_id]

    if direction == 0:
        link_id = pois_extract.type_ID.to_list()[0]
    else:
        link_id = pois_extract.type_ID.to_list()[0]
    connected_edges = []
    if link_id >= 0:
        extract_links = links_gdf[links_gdf.ID == link_id]
        conn_edges = ast.literal_eval(extract_links.conn_edges.to_list()[0])
        connected_edges = conn_edges
        conn_edges.remove(segment_id)

        for kl in conn_edges:
            if link_id == segments_gdf[segments_gdf.ID == kl].link_0.to_list()[0]:
                other_pois_extract = poi_dir_0[poi_dir_0.segment_id == kl]
                tcs.append((kl, 1, other_pois_extract.tc_0.to_list()[0]))
            elif link_id == segments_gdf[segments_gdf.ID == kl].link_1.to_list()[0]:
                other_pois_extract = poi_dir_1[poi_dir_1.segment_id == kl]
                tcs.append((kl, 0, other_pois_extract.tc_0.to_list()[-1]))

    sum_tc = sum([el[-1] for el in tcs])
    return [
        (tcs[ij][0], tcs[ij][1], tcs[ij][2] / sum_tc) for ij in range(0, len(tcs))
    ], connected_edges


def retrieve_demand_split_ratios_for_demand_on_lonely_linkage(
    link_id, links_gdf, segments_gdf, poi_dir_0, poi_dir_1
):
    """
    retrieves the ratios of splitting demand at a linkage
    :param segment_id:
    :param links_df:
    :param pois_df:
    :return:
    """
    tcs = []
    connected_edges = []
    if link_id >= 0:
        extract_links = links_gdf[links_gdf.ID == link_id]
        conn_edges = ast.literal_eval(extract_links.conn_edges.to_list()[0])
        connected_edges = conn_edges

        for kl in conn_edges:
            if link_id == segments_gdf[segments_gdf.ID == kl].link_0.to_list()[0]:
                other_pois_extract = poi_dir_0[poi_dir_0.segment_id == kl]
                tcs.append((kl, 1, other_pois_extract.tc_0.to_list()[0]))
            elif link_id == segments_gdf[segments_gdf.ID == kl].link_1.to_list()[0]:
                other_pois_extract = poi_dir_1[poi_dir_1.segment_id == kl]
                tcs.append((kl, 0, other_pois_extract.tc_0.to_list()[-1]))

    sum_tc = sum([el[-1] for el in tcs])
    return [
        (tcs[ij][0], tcs[ij][1], tcs[ij][2] / sum_tc) for ij in range(0, len(tcs))
    ], connected_edges


def equal_linkages_points(poi_dir_0, poi_dir_1, links_gdf):
    """

    :param model:
    :param poi_dir_0:
    :param poi_dir_1:
    :param links_gdf:
    :param segments_gdf:
    :return: list of touples indicating indices and directions of equalization
    """
    # for each poi in dir
    equal_ids = {}
    link_nodes = []
    link_indices = links_gdf.ID.to_list()
    for kl in range(0, len(links_gdf)):
        equal_vars = []
        link_id = link_indices[kl]
        extract_poi_0 = poi_dir_0[
            (poi_dir_0.type_ID == link_id) & (poi_dir_0.pois_type == "link")
        ]
        extract_poi_1 = poi_dir_1[
            (poi_dir_1.type_ID == link_id) & (poi_dir_1.pois_type == "link")
        ]
        for ind in extract_poi_0.index.to_list():
            equal_vars.append((ind, 0))

        for ind in extract_poi_1.index.to_list():
            equal_vars.append((ind, 1))

        equal_ids[link_id] = equal_vars
        link_nodes = link_nodes + equal_vars
    return equal_ids, link_nodes


def is_linkage(ID, direction, pois_df_0, pois_df_1):
    """

    :param ID: poi ID
    :param direction: 0/1
    :param pois_df_0:
    :param pois_df_1:
    :return: True/False, linkage ID of this Poi (which points to information in link_gdf
    """
    if direction == 0:
        pois_df = pois_df_0
    else:
        pois_df = pois_df_1

    extract = pois_df[pois_df.index == ID]
    if extract.type_ID.to_list()[0] >= 0 and extract.pois_type.to_list()[0] == "link":
        return True, extract.type_ID.to_list()[0]
    else:
        return False, None


def get_segment(ID, direction, pois_df_0, pois_df_1):
    """
    function retrieving the segment ID of the segment the POI lies on
    :param ID:
    :param direction:
    :param pois_df_0:
    :param pois_df_1:
    :return:
    """
    if direction == 0:
        return pois_df_0[pois_df_0.index == ID].segment_id.to_list()[0]
    else:
        return pois_df_1[pois_df_1.index == ID].segment_id.to_list()[0]


def clean_segments(links_gdf, segments_gdf, pois_df):
    to_del = []
    segment_ids = segments_gdf.ID.to_list()
    for s in segment_ids:
        es = pois_df[pois_df.segment_id == s]
        if not any(es.tc_0.array > 0):
            to_del.append(s)

    # clean links_gdf
    nb_of_conn_edges = links_gdf.nb_of_conn_edges.to_list()
    conn_edges = links_gdf.conn_edges.to_list()

    new_conn_edges = []
    new_nb_conn_edges = []

    for ij in range(0, len(conn_edges)):
        n = ast.literal_eval(conn_edges[ij])
        new_n = n
        new_nb = nb_of_conn_edges[ij]
        for kl in n:
            if kl in to_del:
                new_nb = new_nb - 1
                new_n.remove(kl)
        new_conn_edges.append(str(new_n))
        new_nb_conn_edges.append(new_nb)

    links_gdf["nb_of_conn_edges"] = new_nb_conn_edges
    links_gdf["conn_edges"] = new_conn_edges

    # clean segments_gdf
    segments_gdf = segments_gdf[~segments_gdf.ID.isin(to_del)]

    # clean pois
    pois_df = pois_df[~pois_df.segment_id.isin(to_del)]
    pois_df.ID = range(0, len(pois_df))
    pois_df = pois_df.set_index("ID")

    return pois_df, segments_gdf, links_gdf


def filter_segments(pois_df, segments_gdf, links_gdf, keep_segment):
    """
    filter all information to keep only segments with seg_id in keep_segment
    :param pois_df:
    :param segments_gdf:
    :param links_gdf:
    :param keep_segment:
    :return:
    """

    segments_gdf = segments_gdf[segments_gdf.ID.isin(keep_segment)]
    pois_df = pois_df[pois_df.segment_id.isin(keep_segment)]

    left_links = segments_gdf.link_0.to_list() + segments_gdf.link_1.to_list()
    links_gdf = links_gdf[links_gdf.ID.isin(left_links)]

    return pois_df, segments_gdf, links_gdf


def are_ras_along_the_way(
    ij,
    seg_id,
    direction,
    path,
    pois_df_0,
    pois_df_1,
    kl,
    end_seg_id,
    end_direction,
    n0,
    n1,
    segments_gdf,
):
    ras_appears = []
    factor = 1

    # flow just takes place on current_segment
    if seg_id == end_seg_id:
        if direction == 0:
            seg_extract = pois_df_0[pois_df_0.segment_id == seg_id]
        else:
            seg_extract = pois_df_1[pois_df_1.segment_id == seg_id]

        indices = seg_extract.index.to_list()
        ind_0 = indices.index(ij)
        ind_1 = indices.index(kl)
        seg_extract = seg_extract[seg_extract.index.isin(indices[ind_0 : (ind_1 + 1)])]

        if "ra" not in seg_extract.pois_type.to_list():
            ras_appears.append(False)
        else:
            ras_appears.append(True)

    else:
        # I need to trace on the path

        if direction == 0 and end_direction == 1:
            kl = kl - n0
        elif direction == 1 and end_direction == 0:
            kl = kl - n1

        # check position of source node on seg_id -> is there a ra along the way?
        if direction == 0:
            seg_extract = pois_df_0[pois_df_0.segment_id == seg_id]
        else:
            seg_extract = pois_df_1[pois_df_1.segment_id == seg_id]

        indices = seg_extract.index.to_list()
        ind = indices.index(ij)
        seg_extract = seg_extract[seg_extract.index.isin(indices[ind : len(indices)])]

        if "ra" not in seg_extract.pois_type.to_list():
            ras_appears.append(False)
        else:
            ras_appears.append(True)
        # TODO: change "children"-based rationing to "parent"-based
        #   -> for each ancestor -> parent ->
        if len(path[str(seg_id) + "_" + str(direction) + "_0"].children) > 0:
            segs = [
                p.name[0]
                for p in path[str(seg_id) + "_" + str(direction) + "_0"].children
            ]
            dirs = [
                p.name[1]
                for p in path[str(seg_id) + "_" + str(direction) + "_0"].children
            ]
            # factor = factor * (1 / len(path[str(seg_id) + '_' + str(direction) + '_0'].children))
            # factor = factor *

        # check segments in between
        # find end_seg node
        name = (end_seg_id, end_direction)
        end_node = None
        for d in path[str(seg_id) + "_" + str(direction) + "_0"].descendants:
            if d.name == name:
                end_node = d
                break
        prev_seg = seg_id
        prev_dir = direction
        if end_node is not None:
            ancestors = end_node.ancestors
            filtered_ancestors = [
                node for node in ancestors if not node.name == (seg_id, direction)
            ]

            if not len(filtered_ancestors) == 0:
                for a in filtered_ancestors:
                    s = a.name[0]
                    d = a.name[1]

                    if d == 0:
                        poi_extract = pois_df_0[pois_df_0.segment_id == s]
                    else:
                        poi_extract = pois_df_1[pois_df_1.segment_id == s]

                    if not "ra" in poi_extract.pois_type.to_list():
                        ras_appears.append(False)
                    else:
                        ras_appears.append(True)

                    siblings = a.parent.children
                    filtered_siblings = [a.name] + [
                        s.name for s in siblings if s is not a
                    ]
                    # collect all traffic_flow numbers for each same_o
                    trafficflow_numbers = [
                        get_traffic_count(
                            filtered_siblings[ij][0],
                            filtered_siblings[ij][1],
                            pois_df_0,
                            pois_df_1,
                        )
                        for ij in range(0, len(filtered_siblings))
                    ]
                    # if len(a.children) > 0:
                    factor = factor * round(
                        (trafficflow_numbers[0] / sum(trafficflow_numbers)), 1
                    )

        # check ending segment
        prev_seg = end_node.parent.name[0]
        prev_dir = end_node.parent.name[1]
        if end_direction == 0:
            end_extract = pois_df_0[pois_df_0.segment_id == end_seg_id]
        else:
            end_extract = pois_df_1[pois_df_1.segment_id == end_seg_id]

        if prev_dir == 0:
            link_id = segments_gdf[segments_gdf.ID == prev_seg].link_1.to_list()[0]
        else:
            link_id = segments_gdf[segments_gdf.ID == prev_seg].link_0.to_list()[0]
        if (
            end_direction == 0
            and link_id
            == segments_gdf[segments_gdf.ID == end_seg_id].link_0.to_list()[0]
        ):
            end_extract = end_extract.sort_values(
                by="dist_along_segment", ascending=False
            )
        elif (
            end_direction == 1
            and link_id
            == segments_gdf[segments_gdf.ID == end_seg_id].link_1.to_list()[0]
        ):
            end_extract = end_extract.sort_values(by="dist_along_segment")

        indices = end_extract.index.to_list()
        ind = indices.index(kl)
        end_extract = end_extract[end_extract.index.isin(indices[ind : len(indices)])]

        if not "ra" in end_extract.pois_type.to_list():
            ras_appears.append(False)
        else:
            ras_appears.append(True)

        siblings = end_node.parent.children
        filtered_siblings = [end_node.name] + [
            s.name for s in siblings if s is not end_node
        ]
        # collect all traffic_flow numbers for each same_o
        trafficflow_numbers = [
            get_traffic_count(
                filtered_siblings[ij][0], filtered_siblings[ij][1], pois_df_0, pois_df_1
            )
            for ij in range(0, len(filtered_siblings))
        ]
        # if len(a.children) > 0:
        factor = factor * round((trafficflow_numbers[0] / sum(trafficflow_numbers)), 1)

    if True in ras_appears:
        return True, factor
    else:
        return False, factor


def get_traffic_count(seg_id, direction, pois_0, pois_1):
    """
    getting traffic count
    :param seg_id:
    :param direction:
    :param pois_0:
    :param pois_1:
    :return:
    """
    if direction == 0:
        return pois_0[pois_0.segment_id == seg_id]["traffic_flow"].to_list()[0]
    else:
        return pois_1[pois_1.segment_id == seg_id]["traffic_flow"].to_list()[0]


def plot_segments(segments_gdf):
    geoms = segments_gdf.geometry.to_list()
    IDs = segments_gdf.ID.to_list()
    plt.subplots()
    for ij in range(0, len(IDs)):
        g = geoms[ij]
        c = g.centroid
        plt.plot(*g.xy)
        plt.text(c.x, c.y, str(IDs[ij]))


def edit_asfinag_file(filename):

    asfinag_data = pd.read_excel(filename + ".xls", header=2)

    extract_asfinag = asfinag_data[asfinag_data["Unnamed: 0"] == "S01"]
    destination_a = extract_asfinag["Unnamed: 5"].to_list()[0]
    destination_b = extract_asfinag["Unnamed: 5"].to_list()[-7]

    list_ind = asfinag_data["Unnamed: 5"].to_list()

    ind_a = list_ind.index(destination_a)
    ind_b = list_ind.index(destination_b)

    n = len(extract_asfinag)

    asfinag_data.at[ind_a:ind_b, "Unnamed: 0"] = "S1A"
    asfinag_data.at[ind_b : (ind_a + n - 1), "Unnamed: 0"] = "S1B"

    asfinag_data.to_csv(filename + "_edited.csv", index=False)
    return asfinag_data


def create_file_with_maximum_util(folder_file):
    """
    from a folder with multiple .xls-files, this function creates a file with maximum values for each traffic counter
    based on all .xls-files (ASFINAG format)
    :param folder_file: String
    :return: pandas.DataFrame
    """
    # collect all .xls files as dataframes in a list
    # create one reference dataframe where max values are collected
    # for each row of these -> extract from each in list -> get max (iterative -- replace max_val if val > curr_max_val)
    #
    files = [
        f
        for f in listdir(folder_file)
        if isfile(join(folder_file, f)) and (f[-4:] == ".xls" and not f[:4] == "Jahr")
    ]

    processed_files = [
        edit_asfinag_file(folder_file + "/" + f[: len(f) - 4]) for f in files
    ]

    # finding largest file and making it to reference file
    ind_max_len = np.argmax([len(f) for f in processed_files])
    ref_file = processed_files[ind_max_len]
    old_ref_file = ref_file.copy()
    # unique keys: 'Unnamed: 0', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 6'
    highways = ref_file["Unnamed: 0"].to_list()
    locations = ref_file["Unnamed: 2"].to_list()
    traffic_counter_ids = ref_file["Unnamed: 3"].to_list()
    direction = ref_file["Unnamed: 4"].to_list()
    car_type = ref_file["Unnamed: 6"].to_list()
    indices = ref_file.index.to_list()

    for ij in range(0, len(ref_file)):
        hw = highways[ij]
        lc = locations[ij]
        tc = traffic_counter_ids[ij]
        dir = direction[ij]
        ct = car_type[ij]
        ind = indices[ij]

        current_max_val = -1
        for file in processed_files:
            curr_extract_f = file[
                (file["Unnamed: 0"] == hw)
                & (file["Unnamed: 2"] == lc)
                & (file["Unnamed: 3"] == tc)
                & (file["Unnamed: 4"] == dir)
                & (file["Unnamed: 6"] == ct)
            ]
            if len(curr_extract_f) > 0:
                if curr_extract_f["Kfz/24h"].to_list()[0] > current_max_val:
                    current_max_val = curr_extract_f["Kfz/24h"].to_list()[0]

        ref_file.at[ind, "Kfz/24h"] = current_max_val

    file_with_max_vals = ref_file.copy()
    return file_with_max_vals

def edit_asfinag_file(filename):

    asfinag_data = pd.read_excel(filename + ".xls", header=2)

    extract_asfinag = asfinag_data[asfinag_data["Unnamed: 0"] == "S01"]
    destination_a = extract_asfinag["Unnamed: 5"].to_list()[0]
    destination_b = extract_asfinag["Unnamed: 5"].to_list()[-7]

    list_ind = asfinag_data["Unnamed: 5"].to_list()

    ind_a = list_ind.index(destination_a)
    ind_b = list_ind.index(destination_b)

    n = len(extract_asfinag)

    asfinag_data.at[ind_a:ind_b, "Unnamed: 0"] = "S1A"
    asfinag_data.at[ind_b : (ind_a + n - 1), "Unnamed: 0"] = "S1B"

    asfinag_data.to_csv(filename + "_edited.csv", index=False)
    return asfinag_data


def create_file_with_maximum_util(folder_file):
    """
    from a folder with multiple .xls-files, this function creates a file with maximum values for each traffic counter
    based on all .xls-files (ASFINAG format)
    :param folder_file: String
    :return: pandas.DataFrame
    """
    # collect all .xls files as dataframes in a list
    # create one reference dataframe where max values are collected
    # for each row of these -> extract from each in list -> get max (iterative -- replace max_val if val > curr_max_val)
    #
    files = [
        f
        for f in listdir(folder_file)
        if isfile(join(folder_file, f)) and (f[-4:] == ".xls" and not f[:4] == "Jahr")
    ]

    processed_files = [
        edit_asfinag_file(folder_file + "/" + f[: len(f) - 4]) for f in files
    ]

    # finding largest file and making it to reference file
    ind_max_len = np.argmax([len(f) for f in processed_files])
    ref_file = processed_files[ind_max_len]
    old_ref_file = ref_file.copy()
    # unique keys: 'Unnamed: 0', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 6'
    highways = ref_file["Unnamed: 0"].to_list()
    locations = ref_file["Unnamed: 2"].to_list()
    traffic_counter_ids = ref_file["Unnamed: 3"].to_list()
    direction = ref_file["Unnamed: 4"].to_list()
    car_type = ref_file["Unnamed: 6"].to_list()
    indices = ref_file.index.to_list()

    for ij in range(0, len(ref_file)):
        hw = highways[ij]
        lc = locations[ij]
        tc = traffic_counter_ids[ij]
        dir = direction[ij]
        ct = car_type[ij]
        ind = indices[ij]

        current_max_val = -1
        for file in processed_files:
            curr_extract_f = file[
                (file["Unnamed: 0"] == hw)
                & (file["Unnamed: 2"] == lc)
                & (file["Unnamed: 3"] == tc)
                & (file["Unnamed: 4"] == dir)
                & (file["Unnamed: 6"] == ct)
            ]
            if len(curr_extract_f) > 0:
                if curr_extract_f["Kfz/24h"].to_list()[0] > current_max_val:
                    current_max_val = curr_extract_f["Kfz/24h"].to_list()[0]

        ref_file.at[ind, "Kfz/24h"] = current_max_val

    file_with_max_vals = ref_file.copy()
    return file_with_max_vals