def get_children_from_seg(seg_id, prev_name, direc, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax, stop_then=False):
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
                    if link_id == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]:
                        name = str(seg_id) + '_0'
                        seg_tree[name] = Node((c, 0), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(c, name, 0, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax, stop_then=True)
                            else:
                                get_children_from_seg(c, name, 0, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax)

                    elif link_id == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]:
                        name = str(seg_id) + '_1'
                        seg_tree[name] = Node((c, 1), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(c, name, 1, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax,
                                                      stop_then=True)
                            else:
                                get_children_from_seg(c, name, 1, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax)

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
                    if link_id == segments_gdf[segments_gdf.ID == c].link_0.to_list()[0]:
                        name = str(seg_id) + '_0'

                        seg_tree[name] = Node((c, 0), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(c, name, 0, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax,
                                                      stop_then=True)
                            else:
                                get_children_from_seg(c, name, 0, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax)
                    elif link_id == segments_gdf[segments_gdf.ID == c].link_1.to_list()[0]:
                        name = str(seg_id) + '_1'
                        seg_tree[name] = Node((c, 1), parent=seg_tree[prev_name])
                        if not stop_then:
                            if current_dist > dmax:
                                get_children_from_seg(c, name, 1, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax,
                                                      stop_then=True)
                            else:
                                get_children_from_seg(c, name, 1, segments_gdf, links_gdf, pois_df, seg_tree, current_dist, dmax)
    return seg_tree