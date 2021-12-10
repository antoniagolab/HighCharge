import numpy as np

from utils import *
from pyomo.environ import *
from optimization_parameters import *

# was wäre, wenn ich für jeden möglichen Weg eine neure Matrixspalte erstellen würde
# dann müsste ich zuerst einmal ein DF erstellen, in dem ich da
# aber das würd alles um das X-fache komplizierter machen
# kann ich das umgehen, indem ich einfach Segmente identifiziere, wo flows zu ende gehen?
# (1) nach definieren von linkage -> check ob da Flow zu Ende geht auf dem Segment;
# (2)

# Plan: In IO-rels; ending nodes speichern
# für alle ending nodes checken, ob es IO-nodes gibt, die weiter greifen
# wenn das der Fall ist, dann muss der eingeleitete Teil der nicht von dem Segment davor kommt, die Obergrenze für das
# Ouput in dem letzten Teil sein!
# Was für eine Info brauch ich für ending nodes? (prev_segm, prev_dir, last_input_node, last_dir)


def is_linkage(node, poi_dir_0, poi_dir_1, links_gdf):
    equal_ids, link_nodes = equal_linkages_points(poi_dir_0, poi_dir_1, links_gdf)
    if node in link_nodes:
        for k in equal_ids.keys():
            if node in equal_ids[k]:
                return True, k
    else:
        return False, None


def parse_io_rels(list_ios):
    io_nodes = []
    for l in list_ios:
        io_nodes.append(l[1][0])
        io_nodes.append(l[1][1])

    return io_nodes


def append_first_input_output_by_poi_for_seg(
    input_output_relations,
    poi_indices,
    init_direction,
    path,
    current_node,
    pois_df_0,
    pois_df_1,
    segments_gdf,
    current_dists,
    dmax,

):
    """
    TODO: enforce coverage in first link at least, even though distance is limited
    :param input_output_relations:
    :param poi_indices: indices of currently observed pois
    :param init_direction: direction of initial POIS
    :param path:
    :param current_node:
    :param pois_df_0:
    :param pois_df_1:
    :param segments_gdf:
    :param current_dists:
    :param dmax:
    :return:
    """
    segment_id = path[current_node].name[0]
    direction = path[current_node].name[1]
    path_endings = []

    if direction == 0:
        pois_df = pois_df_0
    else:
        pois_df = pois_df_1

    extract_pois = pois_df[pois_df.segment_id == segment_id]
    g = segments_gdf[segments_gdf.ID == segment_id].geometry.to_list()[0]
    g_length = g.length

    if direction == 1:
        extract_pois["dist_along_segment"] = [
            g_length - d for d in extract_pois.dist_along_segment.to_list()
        ]

    extract_pois = extract_pois.sort_values(by="dist_along_segment")
    distances = extract_pois.dist_along_segment.array
    indices = extract_pois.index.array
    poi_ids = extract_pois.index.to_list()

    ranges = [distances[ij + 1] - distances[ij] for ij in range(0, len(indices) - 1)]

    for kl in range(0, len(indices) - 1):
        upcoming_dists = current_dists + ranges[kl]

        for ij in range(0, len(poi_indices)):
            if kl == 0 and ij == 0:
                input_output_relations.append(
                    (
                        (poi_indices[ij], init_direction),
                        ((indices[kl], direction), (indices[kl + 1], direction)),
                    )
                )

            elif (-1e-3 < current_dists[ij] <= dmax) and (
                -1e-3 < upcoming_dists[ij] <= dmax
            ):
                input_output_relations.append(
                    (
                        (poi_indices[ij], init_direction),
                        ((indices[kl], direction), (indices[kl + 1], direction)),
                    )
                )

        current_dists = upcoming_dists

    if all(current_dists > np.array([dmax] * len(current_dists))):

        return input_output_relations, path_endings
    else:
        current_children_to_visit = path[current_node].children
        # need to find keys
        current_children_to_visit_keys = [
            k for k in path.keys() if path[k] in current_children_to_visit
        ]
        for k in current_children_to_visit_keys:
            force_first = True
            input_output_relations, path_endings = append_input_output_by_poi_for_seg(
                input_output_relations,
                indices[-1],
                direction,
                poi_indices,
                init_direction,
                path,
                k,
                pois_df_0,
                pois_df_1,
                segments_gdf,
                current_dists,
                dmax,
                segment_id,
                path_endings,
                force_first
            )
        return input_output_relations, path_endings


def append_input_output_by_poi_for_seg(
    input_output_relations,
    last_index,
    last_direction,
    poi_indices,
    init_direction,
    path,
    current_node,
    pois_df_0,
    pois_df_1,
    segments_gdf,
    current_dists,
    dmax,
    last_seg_id,
    path_endings,
    force_first
):
    """

    :param input_output_relations:
    :param last_index: index of lastly visited poi (link?)
    :param last_direction: direction of lastly visited segment
    :param poi_indices: indices of currently observed pois
    :param init_direction: direction of initial POIS
    :param path:
    :param current_node:
    :param pois_df_0:
    :param pois_df_1:
    :param segments_gdf:
    :param current_dists:
    :param dmax:
    :param last_seg_id:
    :param force_first:
    :return:
    """
    segment_id = path[current_node].name[0]
    direction = path[current_node].name[1]

    if direction == 0:
        pois_df = pois_df_0
    else:
        pois_df = pois_df_1

    extract_pois = pois_df[pois_df.segment_id == segment_id]
    g = segments_gdf[segments_gdf.ID == segment_id].geometry.to_list()[0]
    g_length = g.length

    if direction == 1:
        extract_pois["dist_along_segment"] = [
            g_length - d for d in extract_pois.dist_along_segment.to_list()
        ]

    extract_pois = extract_pois.sort_values(by="dist_along_segment")
    distances = extract_pois.dist_along_segment.array
    indices = extract_pois.index.array
    ranges = [distances[ij + 1] - distances[ij] for ij in range(0, len(indices) - 1)]

    for kl in range(0, len(indices) - 1):

        upcoming_dists = current_dists + ranges[kl]

        # if kl < len(indices) - 2:
        #     dist_after = upcoming_dists + ranges[kl + 1]

        for ij in range(0, len(poi_indices)):
            if force_first and ij == len(poi_indices)-1  and kl == 0:
                input_output_relations.append(
                    (
                        (poi_indices[ij], init_direction),
                        ((indices[kl], direction), (indices[kl + 1], direction)),
                    )
                )
                # if dist_after[ij] >= dmax:
                #     path_endings.append(
                #         (
                #             (poi_indices[ij], init_direction),
                #             (last_seg_id, last_direction, indices[kl + 1], direction),
                #         )
                #     )

            elif (-1e-3 < current_dists[ij] <= dmax) and (
                -1e-3 < upcoming_dists[ij] <= dmax
            ):
                input_output_relations.append(
                    (
                        (poi_indices[ij], init_direction),
                        ((indices[kl], direction), (indices[kl + 1], direction)),
                    )
                )
                # if kl <= len(indices) - 2 and dist_after[ij] >= dmax:
                #     path_endings.append(
                #         (
                #             (poi_indices[ij], init_direction),
                #             (last_seg_id, last_direction, indices[kl + 1], direction),
                #         )
                #     )

        current_dists = upcoming_dists

    if current_dists.all() > np.array([dmax] * len(current_dists)).all():
        return input_output_relations, path_endings
    else:
        current_children_to_visit = path[current_node].children
        # need to find keys
        current_children_to_visit_keys = [
            k for k in path.keys() if path[k] in current_children_to_visit
        ]
        force_first = False
        for k in current_children_to_visit_keys:
            input_output_relations, path_endings = append_input_output_by_poi_for_seg(
                input_output_relations,
                indices[-1],
                direction,
                poi_indices,
                init_direction,
                path,
                k,
                pois_df_0,
                pois_df_1,
                segments_gdf,
                current_dists,
                dmax,
                segment_id,
                path_endings,
                force_first
            )

        return input_output_relations, path_endings


def constraint_input_ouput_relations_for_pois_on_segm(
    segment_id,
    direction,
    pois_df,
    pois_df_0,
    pois_df_1,
    links_df,
    segments_gdf,
    dmax,
):
    """

    :param direction: integer = 0/1
    :param segment_id:
    :param pois_df_0:
    :param pois_df_1:
    :param links_df:
    :param segments_gdf:
    :param dmax:
    :return:
    """
    input_output_relations = []

    name = str(segment_id) + "_" + str(direction) + "_0"
    segm_tree = {name: Node((segment_id, direction))}
    unfiltered_path = get_children_from_seg(
        segment_id,
        name,
        direction,
        segments_gdf,
        links_df,
        pois_df,
        segm_tree,
        0,
        dmax,
        stop_then=False,
    )
    path = filter_path(unfiltered_path, segments_gdf, name)
    # collect pois of segment
    if direction == 0:
        pois_df = pois_df_0
    else:
        pois_df = pois_df_1

    extract_pois = pois_df[pois_df.segment_id == segment_id]
    g = segments_gdf[segments_gdf.ID == segment_id].geometry.to_list()[0]
    g_length = g.length

    if direction == 1:
        extract_pois["dist_along_segment"] = [
            g_length - d for d in extract_pois.dist_along_segment.to_list()
        ]

    extract_pois = extract_pois.sort_values(by="dist_along_segment")
    distances = extract_pois.dist_along_segment.to_list()

    # np.array with current distance counters
    # initializing it at the pois_position 1 (excluding very first link as demand there = 0
    dist_init = 0
    current_dists = np.array([dist_init - d for d in distances])
    indices = extract_pois.index.to_list()

    # for each POI along path, update current_dists and if current_dist >= 1e-3 and current_dist <= dmax, then add
    # touple

    return (
        append_first_input_output_by_poi_for_seg(
            input_output_relations,
            indices,
            direction,
            path,
            name,
            pois_df_0,
            pois_df_1,
            segments_gdf,
            current_dists,
            dmax,
        ),
        path,
    )


def OLD_constraint_input_output(
    model, pois_df_0, pois_df_1, pois_df, links_gdf, segments_gdf, dmax
):

    # TODO: regard linkages!!
    # run for each segment, the path way
    # for each poi on it: gather input-outputs
    # check in input_output-rel if linkage
    # if linkage: tc_ratios needed, based on these define input_output
    # remove an input-output-rel if constrained

    segment_ids = segments_gdf.ID.to_list()
    n0 = len(pois_df_0)
    n1 = len(pois_df_1)

    model.constraint_output_input = ConstraintList()
    for seg_id in segment_ids:

        # dir = 0
        output_0, path_info_0 = constraint_input_ouput_relations_for_pois_on_segm(
            seg_id,
            0,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )

        input_output_rels_0 = output_0[0]
        endings_0 = output_0[1]
        # dir = 1
        output_1, path_info_1 = constraint_input_ouput_relations_for_pois_on_segm(
            seg_id,
            1,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        input_output_rels_1 = output_1[0]
        endings_1 = output_1[1]

        poi_0_extract = pois_df_0[pois_df_0.segment_id == seg_id]
        poi_1_extract = pois_df_1[pois_df_1.segment_id == seg_id]

        poi_0_indices = poi_0_extract.index.to_list()
        poi_1_indices = poi_1_extract.index.to_list()
        for ij in range(0, len(poi_0_extract)):
            i_source_node = poi_0_indices[ij]
            gathered_IO_rels = [
                io for io in input_output_rels_0 if (poi_0_indices[ij], 0) == io[0]
            ]
            model.constraint_output_input.add(
                model.pE_input_0[i_source_node, i_source_node] == 0
            )
            # need to visit linkages first and not visit these later!!
            copied_gathered_IO_rels = gathered_IO_rels.copy()
            for g in gathered_IO_rels:
                # check if second el is linkage
                IO = g[1]
                ouput = IO[0]
                input = IO[1]
                res, link_id = is_linkage(input[0], input[1], pois_df_0, pois_df_1)

                if res:
                    linkages = link_equalizor[link_id]
                    # I need to find all indices describing these and then seek incoming flux and outgoing
                    # based on this I need to evaluate incoming flux

                    # find incoming flux
                    incoming_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][1] == el
                    ]
                    # find outpouring flux
                    outgoing_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][0] == el
                    ]
                    input_dir = {}
                    e = linkages[0]
                    if len(incoming_flux) > 0:
                        incoming_flux_0 = [
                            mn for mn in incoming_flux if mn[1][0][1] == 0
                        ]
                        incoming_flux_1 = [
                            mn for mn in incoming_flux if mn[1][0][1] == 1
                        ]

                        if e[1] == 0:
                            model.constraint_output_input.add(
                                model.pE_input_0[i_source_node, e[0]]
                                == sum(
                                    [
                                        model.pE_output_0[i_source_node, mn[1][0][0]]
                                        for mn in incoming_flux_0
                                    ]
                                )
                                + sum(
                                    [
                                        model.pE_output_0[
                                            i_source_node, mn[1][0][0] + n0
                                        ]
                                        for mn in incoming_flux_1
                                    ]
                                )
                            )
                        else:
                            model.constraint_output_input.add(
                                model.pE_input_0[i_source_node, e[0] + n0]
                                == sum(
                                    [
                                        model.pE_output_0[i_source_node, mn[1][0][0]]
                                        for mn in incoming_flux_0
                                    ]
                                )
                                + sum(
                                    [
                                        model.pE_output_0[
                                            i_source_node, mn[1][0][0] + n0
                                        ]
                                        for mn in incoming_flux_1
                                    ]
                                )
                            )

                        # (from_seg, ID, direction, ratios of belonging segments)
                        # for each incoming flux
                        gathered_incoming_info = []
                        for i in incoming_flux:
                            on_segment = get_segment(
                                i[1][0][0], i[1][0][1], pois_df_0, pois_df_1
                            )
                            ratios, _ = retrieve_demand_split_ratios(
                                on_segment,
                                i[1][0][1],
                                pois_df,
                                links_gdf,
                                segments_gdf,
                                pois_df_0,
                                pois_df_1,
                            )
                            gathered_incoming_info.append(
                                (on_segment, i[1][0][0], i[1][0][1], ratios)
                            )

                        if len(outgoing_flux) > 0:
                            for op in range(0, len(outgoing_flux)):
                                d = outgoing_flux[op][1][1][1]
                                of_id = outgoing_flux[op][1][1][0]
                                on_segment = get_segment(of_id, d, pois_df_0, pois_df_1)
                                if d == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id]
                                        == sum(
                                            [
                                                mn[2]
                                                * model.pE_output_0[i_source_node, l[1]]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 0 and mn[0] == on_segment
                                            ]
                                        )
                                        + sum(
                                            [
                                                mn[2]
                                                * model.pE_output_0[
                                                    i_source_node, l[1] + n0
                                                ]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 1 and mn[0] == on_segment
                                            ]
                                        )
                                    )
                                else:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id + n0]
                                        == sum(
                                            [
                                                mn[2]
                                                * model.pE_output_0[
                                                    i_source_node, l[1] + n0
                                                ]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 0 and mn[0] == on_segment
                                            ]
                                        )
                                        + sum(
                                            [
                                                mn[2]
                                                * model.pE_output_0[
                                                    i_source_node, l[1] + n0
                                                ]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 1 and mn[0] == on_segment
                                            ]
                                        )
                                    )
                    else:
                        if (len(outgoing_flux)) > 0:

                            (
                                ratio,
                                _,
                            ) = retrieve_demand_split_ratios_for_demand_on_lonely_linkage(
                                link_id, links_gdf, segments_gdf, pois_df_0, pois_df_1
                            )

                            gathered_outgoing_info = []
                            for i in outgoing_flux:
                                on_segment = get_segment(
                                    i[1][1][0], i[1][1][1], pois_df_0, pois_df_1
                                )
                                r = [mn for mn in ratio if mn[0] == on_segment]
                                gathered_outgoing_info.append(
                                    (on_segment, i[1][1][0], i[1][1][1], r)
                                )

                                of_id = i[1][1][0]
                                d = i[1][1][1]
                                if d == 0 and e[1] == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id]
                                        == model.pE_output_0[i_source_node, e[0]] * r[2]
                                    )

                                elif d == 1 and e[1] == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id + n0]
                                        == model.pE_output_0[i_source_node, e[0]] * r[2]
                                    )
                                elif d == 1 and e[1] == 1:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id + n0]
                                        == model.pE_output_0[i_source_node, e[0] + n0]
                                        * r[2]
                                    )

                                else:
                                    model.constraint_output_input.add(
                                        model.pE_input_0[i_source_node, of_id]
                                        == model.pE_output_0[i_source_node, e[0] + n0]
                                        * r[2]
                                    )

                    for i in incoming_flux:
                        gathered_IO_rels.remove(i)
                    for o in outgoing_flux:
                        gathered_IO_rels.remove(o)

            for element in gathered_IO_rels:  # currently visited node is no linkage
                prev_id = element[1][0][0]
                prev_dir = element[1][0][1]
                next_id = element[1][1][0]
                next_dir = element[1][1][1]

                if prev_dir == 0:
                    model.constraint_output_input.add(
                        model.pE_output_0[i_source_node, prev_id]
                        == model.pE_input_0[i_source_node, next_id]
                    )
                else:
                    model.constraint_output_input.add(
                        model.pE_output_0[i_source_node, prev_id + n0]
                        == model.pE_input_0[i_source_node, next_id + n0]
                    )
            # retrieve belonging endings
            # check if there are following I0-relations
            # if yes -> then constraint the ending node by model.pE_output[iii] >= energy_to_link which has not
            # been from the previously travelled segment
            belonging_endings = [
                ending for ending in endings_0 if ending[0] == (i_source_node, 0)
            ]
            for b in belonging_endings:
                b_dir = b[1][3]
                b_id = b[1][2]
                b_seg = get_segment(b_id, b_dir, pois_df_0, pois_df_1)
                prev_seg = b[1][0]
                prev_dir = b[1][1]
                possible_add_IOs = [
                    ad for ad in copied_gathered_IO_rels if ad[1][0] == (b_id, b_dir)
                ]

                if len(possible_add_IOs) > 0:
                    # evaluate the ratio at last linkage which does not stem from b_seg, b_dir
                    if b_dir == 0:
                        extract_curr_seg = pois_df_0[pois_df_0.segment_id == b_seg]
                        link_id = extract_curr_seg.type_ID.to_list()[0]
                    else:
                        extract_curr_seg = pois_df_1[pois_df_1.segment_id == b_seg]
                        link_id = extract_curr_seg.type_ID.to_list()[-1]
                    linkages = link_equalizor[link_id]
                    # find incoming flux
                    incoming_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][1] == el
                    ]
                    # find outpouring flux
                    outgoing_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][0] == el
                    ]
                    # (from_seg, ID, direction, ratios of belonging segments)
                    # for each incoming flux
                    gathered_incoming_info = []
                    for i in incoming_flux:
                        on_segment = get_segment(
                            i[1][0][0], i[1][0][1], pois_df_0, pois_df_1
                        )
                        if not on_segment == prev_seg:
                            ratios, _ = retrieve_demand_split_ratios(
                                on_segment,
                                i[1][0][1],
                                pois_df,
                                links_gdf,
                                segments_gdf,
                                pois_df_0,
                                pois_df_1,
                            )
                            gathered_incoming_info.append(
                                (on_segment, i[1][0][0], i[1][0][1], ratios)
                            )

                    if b_dir == 0:
                        model.constraint_output_input.add(
                            model.pE_output_0[i_source_node, b_id]
                            <= sum(
                                [
                                    mn[2] * model.pE_output_0[i_source_node, l[1]]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 0 and mn[0] == b_seg
                                ]
                            )
                            + sum(
                                [
                                    mn[2] * model.pE_output_0[i_source_node, l[1] + n0]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 1 and mn[0] == b_seg
                                ]
                            )
                        )
                    else:
                        model.constraint_output_input.add(
                            model.pE_input_0[i_source_node, b_id + n0]
                            <= sum(
                                [
                                    mn[2] * model.pE_output_0[i_source_node, l[1]]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 0 and mn[0] == b_seg
                                ]
                            )
                            + sum(
                                [
                                    mn[2] * model.pE_output_0[i_source_node, l[1] + n0]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 1 and mn[0] == b_seg
                                ]
                            )
                        )
                else:
                    if b_dir == 0:
                        model.constraint_output_input.add(
                            model.pE_output_0[i_source_node, b_id] == 0
                        )
                    else:
                        model.constraint_output_input.add(
                            model.pE_output_0[i_source_node, b_id + n0] == 0
                        )

        for ij in range(0, len(poi_1_extract)):
            i_source_node = poi_1_indices[ij]
            gathered_IO_rels = [
                io for io in input_output_rels_1 if (i_source_node, 1) == io[0]
            ]
            model.constraint_output_input.add(
                model.pE_input_1[i_source_node, i_source_node] == 0
            )
            copied_gathered_IO_rels = gathered_IO_rels.copy()
            # need to visit linkages first and not visit these later!!

            for g in gathered_IO_rels:
                # check if second el is linkage
                IO = g[1]
                ouput = IO[0]
                input = IO[1]
                res, link_id = is_linkage(input[0], input[1], pois_df_0, pois_df_1)
                if res:
                    # I need to find all indices describing these and then seek incoming flux and outgoing
                    # based on this I need to evaluate incoming flux
                    linkages = link_equalizor[link_id]

                    # find incoming flux
                    incoming_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][1] == el
                    ]
                    # find outpouring flux
                    outgoing_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][0] == el
                    ]
                    input_dir = {}
                    e = linkages[0]
                    if len(incoming_flux) > 0:
                        incoming_flux_0 = [
                            mn for mn in incoming_flux if mn[1][0][1] == 0
                        ]
                        incoming_flux_1 = [
                            mn for mn in incoming_flux if mn[1][0][1] == 1
                        ]

                        if e[1] == 0:
                            model.constraint_output_input.add(
                                model.pE_input_0[i_source_node, e[0]]
                                == sum(
                                    [
                                        model.pE_output_1[
                                            i_source_node, mn[1][0][0] + n1
                                        ]
                                        for mn in incoming_flux_0
                                    ]
                                )
                                + sum(
                                    [
                                        model.pE_output_1[i_source_node, mn[1][0][0]]
                                        for mn in incoming_flux_1
                                    ]
                                )
                            )
                        else:
                            model.constraint_output_input.add(
                                model.pE_input_1[i_source_node, e[0]]
                                == sum(
                                    [
                                        model.pE_output_1[
                                            i_source_node, mn[1][0][0] + n1
                                        ]
                                        for mn in incoming_flux_0
                                    ]
                                )
                                + sum(
                                    [
                                        model.pE_output_1[i_source_node, mn[1][0][0]]
                                        for mn in incoming_flux_1
                                    ]
                                )
                            )

                        # (from_seg, ID, direction, ratios of belonging segments)
                        # for each incoming flux
                        gathered_incoming_info = []
                        for i in incoming_flux:
                            on_segment = get_segment(
                                i[1][0][0], i[1][0][1], pois_df_0, pois_df_1
                            )
                            ratios, _ = retrieve_demand_split_ratios(
                                on_segment,
                                i[1][0][1],
                                pois_df,
                                links_gdf,
                                segments_gdf,
                                pois_df_0,
                                pois_df_1,
                            )
                            gathered_incoming_info.append(
                                (on_segment, i[1][0][0], i[1][0][1], ratios)
                            )

                        if len(outgoing_flux) > 0:
                            for op in range(0, len(outgoing_flux)):
                                d = outgoing_flux[op][1][1][1]
                                of_id = outgoing_flux[op][1][1][0]
                                on_segment = get_segment(of_id, d, pois_df_0, pois_df_1)
                                if d == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id + n1]
                                        == sum(
                                            [
                                                mn[2]
                                                * model.pE_output_1[
                                                    i_source_node, l[1] + n1
                                                ]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 0 and mn[0] == on_segment
                                            ]
                                        )
                                        + sum(
                                            [
                                                mn[2]
                                                * model.pE_output_1[i_source_node, l[1]]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 1 and mn[0] == on_segment
                                            ]
                                        )
                                    )
                                else:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id]
                                        == sum(
                                            [
                                                mn[2]
                                                * model.pE_output_1[
                                                    i_source_node, l[1] + n1
                                                ]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 0 and mn[0] == on_segment
                                            ]
                                        )
                                        + sum(
                                            [
                                                mn[2]
                                                * model.pE_output_1[i_source_node, l[1]]
                                                for l in gathered_incoming_info
                                                for mn in l[3]
                                                if l[2] == 1 and mn[0] == on_segment
                                            ]
                                        )
                                    )
                    else:
                        if (len(outgoing_flux)) > 0:

                            (
                                ratio,
                                _,
                            ) = retrieve_demand_split_ratios_for_demand_on_lonely_linkage(
                                link_id, links_gdf, segments_gdf, pois_df_0, pois_df_1
                            )

                            gathered_outgoing_info = []
                            for i in outgoing_flux:
                                on_segment = get_segment(
                                    i[1][1][0], i[1][1][1], pois_df_0, pois_df_1
                                )
                                r = [mn for mn in ratio if mn[0] == on_segment]
                                gathered_outgoing_info.append(
                                    (on_segment, i[1][1][0], i[1][1][1], r)
                                )

                                of_id = i[1][1][0]
                                d = i[1][1][1]
                                if d == 0 and e[1] == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id + n1]
                                        == model.pE_output_1[i_source_node, e[0] + n1]
                                        * r[2]
                                    )

                                elif d == 1 and e[1] == 0:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id]
                                        == model.pE_output_1[i_source_node, e[0] + n1]
                                        * r[2]
                                    )
                                elif d == 1 and e[1] == 1:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id]
                                        == model.pE_output_1[i_source_node, e[0]] * r[2]
                                    )

                                else:
                                    model.constraint_output_input.add(
                                        model.pE_input_1[i_source_node, of_id + n1]
                                        == model.pE_output_1[i_source_node, e[0]] * r[2]
                                    )

                    for i in incoming_flux:
                        gathered_IO_rels.remove(i)
                    for o in outgoing_flux:
                        gathered_IO_rels.remove(o)

            for element in gathered_IO_rels:  # currently visited node is no linkage
                prev_id = element[1][0][0]
                prev_dir = element[1][0][1]
                next_id = element[1][1][0]

                if prev_dir == 0:
                    model.constraint_output_input.add(
                        model.pE_output_1[i_source_node, prev_id + n1]
                        == model.pE_input_1[i_source_node, next_id + n1]
                    )
                else:
                    model.constraint_output_input.add(
                        model.pE_output_1[i_source_node, prev_id]
                        == model.pE_input_1[i_source_node, next_id]
                    )
            belonging_endings = [
                ending for ending in endings_1 if ending[0] == (i_source_node, 1)
            ]
            for b in belonging_endings:
                b_dir = b[1][3]
                b_id = b[1][2]
                b_seg = get_segment(b_id, b_dir, pois_df_0, pois_df_1)
                prev_seg = b[1][0]
                prev_dir = b[1][1]
                possible_add_IOs = [
                    ad for ad in copied_gathered_IO_rels if ad[1][0] == (b_id, b_dir)
                ]

                if len(possible_add_IOs) > 0:
                    # evaluate the ratio at last linkage which does not stem from b_seg, b_dir
                    if b_dir == 0:
                        extract_curr_seg = pois_df_0[pois_df_0.segment_id == b_seg]
                        link_id = extract_curr_seg.type_ID.to_list()[0]
                    else:
                        extract_curr_seg = pois_df_1[pois_df_1.segment_id == b_seg]
                        link_id = extract_curr_seg.type_ID.to_list()[-1]
                    linkages = link_equalizor[link_id]
                    # find incoming flux
                    incoming_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][1] == el
                    ]
                    # find outpouring flux
                    outgoing_flux = [
                        mn
                        for mn in gathered_IO_rels
                        for el in linkages
                        if mn[1][0] == el
                    ]
                    # (from_seg, ID, direction, ratios of belonging segments)
                    # for each incoming flux
                    gathered_incoming_info = []
                    for i in incoming_flux:
                        on_segment = get_segment(
                            i[1][0][0], i[1][0][1], pois_df_0, pois_df_1
                        )
                        if not on_segment == prev_seg:
                            ratios, _ = retrieve_demand_split_ratios(
                                on_segment,
                                i[1][0][1],
                                pois_df,
                                links_gdf,
                                segments_gdf,
                                pois_df_0,
                                pois_df_1,
                            )
                            gathered_incoming_info.append(
                                (on_segment, i[1][0][0], i[1][0][1], ratios)
                            )

                    if b_dir == 0:
                        model.constraint_output_input.add(
                            model.pE_output_1[i_source_node, b_id + n1]
                            <= sum(
                                [
                                    mn[2] * model.pE_output_1[i_source_node, l[1] + n1]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 0 and mn[0] == b_seg
                                ]
                            )
                            + sum(
                                [
                                    mn[2] * model.pE_output_1[i_source_node, l[1]]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 1 and mn[0] == b_seg
                                ]
                            )
                        )
                    else:
                        model.constraint_output_input.add(
                            model.pE_input_1[i_source_node, b_id]
                            <= sum(
                                [
                                    mn[2] * model.pE_output_1[i_source_node, l[1] + n1]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 0 and mn[0] == b_seg
                                ]
                            )
                            + sum(
                                [
                                    mn[2] * model.pE_output_1[i_source_node, l[1]]
                                    for l in gathered_incoming_info
                                    for mn in l[3]
                                    if l[2] == 1 and mn[0] == b_seg
                                ]
                            )
                        )
                else:
                    if b_dir == 0:
                        model.constraint_output_input.add(
                            model.pE_output_1[i_source_node, b_id + n1] == 0
                        )
                    else:
                        model.constraint_output_input.add(
                            model.pE_output_1[i_source_node, b_id] == 0
                        )


def constraint_equalizing_station_dirs():
    return None


def constraint_Y_i(model, pois_df_0, pois_df_1):
    """
    defining constraints to avoid charging stations being build at nodes representing intersections
    :param model:
    :param pois_df_0:
    :param pois_df_1:
    :return:
    """

    Y_0 = np.zeros([n0, 1])
    Y_1 = np.zeros([n1, 1])
    # constraint first pY_dir_0,
    model.constraint_Y_i = ConstraintList()
    col_poi_type = "pois_type"
    link_type_0 = pois_df_0[col_poi_type].to_list()

    for ij in range(0, n0):
        if link_type_0[ij] == "link":
            model.constraint_Y_i.add(model.pYi_dir_0[ij] == 0)
            Y_0[ij] = 0
        else:
            Y_0[ij] = 1
    # then pY_dir_1,
    link_type_1 = pois_df_1[col_poi_type].to_list()
    for ij in range(0, n1):
        if link_type_1[ij] == "link":
            model.constraint_Y_i.add(model.pYi_dir_1[ij] == 0)
            Y_1[ij] = 0
        else:
            Y_1[ij] = 1


def constraint_charged_input_output(
    model, pois_df_0, pois_df_1, pois_df, links_gdf, segments_gdf, dmax
):

    model.constraint_charged_input_output = ConstraintList()
    seg_ids = segments_gdf.ID.to_list()

    linkages_dir, _ = equal_linkages_points(pois_df_0, pois_df_1, links_gdf)

    # dir = 0
    charged_0 = np.zeros([n0, n0 + n1])
    input_0 = np.zeros([n0, n0 + n1])
    output_0 = np.zeros([n0, n0 + n1])

    charged_1 = np.zeros([n1, n0 + n1])

    counter = 0
    counter2 = 0
    for s in seg_ids:
        output_0, path_info_0 = constraint_input_ouput_relations_for_pois_on_segm(
            s,
            0,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        io_rels_0 = output_0[0]
        output_1, path_info_1 = constraint_input_ouput_relations_for_pois_on_segm(
            s,
            1,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        io_rels_1 = output_1[0]
        poi_0_extract = pois_df_0[pois_df_0.segment_id == s]
        poi_1_extract = pois_df_1[pois_df_1.segment_id == s]

        poi_0_total_inds = pois_df_0.index.to_list()
        poi_1_total_inds = pois_df_1.index.to_list()

        poi_0_indices = poi_0_extract.index.to_list()
        poi_1_indices = poi_1_extract.index.to_list()

        poi_type_0 = pois_df_0[col_poi_type].to_list()
        poi_type_1 = pois_df_1[col_poi_type].to_list()

        for kl in range(0, len(poi_0_indices)):
            i_source_node = poi_0_indices[kl]
            gathered_io_rels = [io for io in io_rels_0 if (i_source_node, 0) == io[0]]
            parsed_rels = parse_io_rels(gathered_io_rels)
            are_linkages = [
                is_linkage(node, pois_df_0, pois_df_1, links_gdf)[0]
                for node in parsed_rels
            ]
            for ij in range(0, len(are_linkages)):
                if are_linkages[ij]:
                    link_id = is_linkage(
                        parsed_rels[ij], pois_df_0, pois_df_1, links_gdf
                    )[1]
                    parsed_rels = parsed_rels + linkages_dir[link_id]
            collected_0 = [io[0] for io in parsed_rels if io[1] == 0]
            collected_1 = [io[0] for io in parsed_rels if io[1] == 1]

            collected_0.append(i_source_node)

            collected_0 = list(set(collected_0))
            collected_1 = list(set(collected_1))

            for ij in model.IDX_0:
                curr_int = poi_0_total_inds[ij]
                if not curr_int in collected_0:
                    model.constraint_charged_input_output.add(
                        model.pE_charged_0[counter + kl, ij] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_input_0[counter + kl, ij] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_output_0[counter + kl, ij] == 0
                    )
                    charged_0[counter + kl, ij] = 2
                else:
                    charged_0[counter + kl, ij] = 1

            for ij in model.IDX_1:
                curr_int = poi_1_total_inds[ij]
                if not curr_int in collected_1:
                    model.constraint_charged_input_output.add(
                        model.pE_charged_0[counter + kl, ij + n0] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_input_0[counter + kl, ij + n0] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_output_0[counter + kl, ij + n0] == 0
                    )
                    charged_0[counter + kl, ij + n0] = 2
                else:
                    charged_0[counter + kl, ij + n0] = 1

        counter = counter + len(poi_0_indices)

        for kl in range(0, len(poi_1_indices)):
            i_source_node = poi_1_indices[kl]
            gathered_io_rels = [io for io in io_rels_1 if (i_source_node, 1) == io[0]]
            parsed_rels = parse_io_rels(gathered_io_rels)
            are_linkages = [
                is_linkage(node, pois_df_0, pois_df_1, links_gdf)[0]
                for node in parsed_rels
            ]
            for ij in range(0, len(are_linkages)):
                if are_linkages[ij]:
                    link_id = is_linkage(
                        parsed_rels[ij], pois_df_0, pois_df_1, links_gdf
                    )[1]
                    parsed_rels = parsed_rels + linkages_dir[link_id]
            collected_0 = [io[0] for io in parsed_rels if io[1] == 0]
            collected_1 = [io[0] for io in parsed_rels if io[1] == 1]

            collected_1.append(i_source_node)

            collected_0 = list(set(collected_0))
            collected_1 = list(set(collected_1))

            for ij in model.IDX_0:
                curr_int = poi_0_total_inds[ij]
                if not curr_int in collected_0:
                    model.constraint_charged_input_output.add(
                        model.pE_charged_1[counter2 + kl, ij + n1] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_input_1[counter2 + kl, ij + n1] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_output_1[counter2 + kl, ij + n1] == 0
                    )
                    charged_1[counter2 + kl, ij + n1] = 2
                else:
                    charged_1[counter2 + kl, ij + n1] = 1

            for ij in model.IDX_1:
                curr_int = poi_1_total_inds[ij]
                if not curr_int in collected_1:
                    model.constraint_charged_input_output.add(
                        model.pE_charged_1[counter2 + kl, ij] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_input_1[counter2 + kl, ij] == 0
                    )
                    model.constraint_charged_input_output.add(
                        model.pE_output_1[counter2 + kl, ij] == 0
                    )
                    charged_1[counter2 + kl, ij] = 2
                else:
                    charged_1[counter2 + kl, ij] = 1

        counter2 = counter2 + len(poi_1_indices)


def constraint_equal_nodes_links(model, poi_dir_0, poi_dir_1, links_gdf):
    link_dir, _ = equal_linkages_points(poi_dir_0, poi_dir_1, links_gdf)

    model.constraint_equal_nodes_links = ConstraintList()

    poi_0_indices = poi_dir_0.index.to_list()
    poi_1_indices = poi_dir_1.index.to_list()

    charged_0 = np.zeros([n0, n0 + n1])
    charged_1 = np.zeros([n1, n0 + n1])
    for k in link_dir.keys():
        vals = link_dir[k]
        #  0 and 1 matrix
        # for each row

        # matrix dir = 0
        for kl in range(0, n0):
            for ij in range(1, len(vals)):
                prev_dir = vals[ij - 1][1]
                next_dir = vals[ij][1]
                if prev_dir == 0:
                    prev_ind = poi_0_indices.index(vals[ij - 1][0])
                else:
                    prev_ind = poi_1_indices.index(vals[ij - 1][0])

                if next_dir == 0:
                    next_ind = poi_0_indices.index(vals[ij][0])
                else:
                    next_ind = poi_1_indices.index(vals[ij][0])

                if prev_dir == 0 and next_dir == 0:
                    charged_0[kl, prev_ind] = 1
                    charged_0[kl, next_ind] = 1
                    charged_1[kl, n1 + prev_ind] = 1
                    charged_1[kl, n1 + next_ind] = 1
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_0[kl, prev_ind] == model.pE_input_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_0[kl, prev_ind]
                        == model.pE_output_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_0[kl, prev_ind]
                        == model.pE_charged_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_1[kl, n1 + prev_ind]
                        == model.pE_input_1[kl, n1 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_1[kl, n1 + prev_ind]
                        == model.pE_output_1[kl, n1 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_1[kl, n1 + prev_ind]
                        == model.pE_charged_1[kl, n1 + next_ind]
                    )
                elif prev_dir == 0 and next_dir == 1:
                    charged_0[kl, prev_ind] = 1
                    charged_0[kl, n0 + next_ind] = 1
                    charged_1[kl, n1 + prev_ind] = 1
                    charged_1[kl, next_ind] = 1
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_0[kl, prev_ind]
                        == model.pE_input_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_0[kl, prev_ind]
                        == model.pE_output_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_0[kl, prev_ind]
                        == model.pE_charged_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_1[kl, n1 + prev_ind]
                        == model.pE_input_1[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_1[kl, n1 + prev_ind]
                        == model.pE_output_1[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_1[kl, n1 + prev_ind]
                        == model.pE_charged_1[kl, next_ind]
                    )
                elif prev_dir == 1 and next_dir == 0:
                    charged_0[kl, n0 + prev_ind] = 1
                    charged_0[kl, next_ind] = 1
                    charged_1[kl, prev_ind] = 1
                    charged_1[kl, n1 + next_ind] = 1
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_0[kl, n0 + prev_ind]
                        == model.pE_input_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_0[kl, n0 + prev_ind]
                        == model.pE_output_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_0[kl, n0 + prev_ind]
                        == model.pE_charged_0[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_1[kl, prev_ind]
                        == model.pE_input_1[kl, n1 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_1[kl, prev_ind]
                        == model.pE_output_1[kl, n1 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_1[kl, prev_ind]
                        == model.pE_charged_1[kl, n1 + next_ind]
                    )
                else:
                    charged_0[kl, n0 + prev_ind] = 1
                    charged_0[kl, n0 + next_ind] = 1
                    charged_1[kl, prev_ind] = 1
                    charged_1[kl, next_ind] = 1
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_0[kl, n0 + prev_ind]
                        == model.pE_input_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_0[kl, n0 + prev_ind]
                        == model.pE_output_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_0[kl, n0 + prev_ind]
                        == model.pE_charged_0[kl, n0 + next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_input_1[kl, prev_ind] == model.pE_input_1[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_output_1[kl, prev_ind]
                        == model.pE_output_1[kl, next_ind]
                    )
                    model.constraint_equal_nodes_links.add(
                        model.pE_charged_1[kl, prev_ind]
                        == model.pE_charged_1[kl, next_ind]
                    )


def constraint_coverage_of_energy_demand(model, energy_demand_matrix_0, energy_demand_matrix_1):
    model.constraint_coverage_of_energy_demand = ConstraintList()

    # dir = 0
    for ij in model.IDX_0:
        model.constraint_coverage_of_energy_demand.add(
            sum([model.pE_charged_0[ij, kl] for kl in model.IDX_0])
            == sum(
                energy_demand_matrix_0[ij, ]
            )
        )

    # dir = 1
    for ij in model.IDX_1:
        model.constraint_coverage_of_energy_demand.add(
            sum([model.pE_charged_1[ij, kl] for kl in model.IDX_1])
            == sum(
                energy_demand_matrix_1[
                    ij,
                ]
            )
        )


def constraint_rel_dem_i_o_charge(model, energy_demand_matrix_0, energy_demand_matrix_1, pois_df_0, pois_df_1, links_gdf):
    model.constraint_i_o_charge = ConstraintList()

    link_directory, link_nodes = equal_linkages_points(pois_df_0, pois_df_1, links_gdf)

    # dir = 0
    for ij in model.IDX_0:
        for kl in model.IDX_3:
            model.constraint_i_o_charge.add(
                model.pE_charged_0[ij, kl]
                - energy_demand_matrix_0[ij, kl]
                - model.pE_input_0[ij, kl]
                + model.pE_output_0[ij, kl]
                + model.test_var_0[ij, kl]
                == 0
            )

    # dir = 1
    for ij in model.IDX_1:
        for kl in model.IDX_3:
            model.constraint_i_o_charge.add(
                model.pE_charged_1[ij, kl]
                - energy_demand_matrix_1[ij, kl]
                - model.pE_input_1[ij, kl]
                + model.pE_output_1[ij, kl]
                + model.test_var_1[ij, kl]
                == 0
            )


def constraint_CS_capacity(model, pois_0, pois_1, pois, energy):

    model.constraint_CS_capacity = ConstraintList()
    col_poi_type = "pois_type"
    col_type_ID = "type_ID"
    g = 1000
    type_IDs = pois[col_type_ID].to_list()
    types = pois[col_poi_type].to_list()

    # find corresponding POI via type_ID
    for ij in model.IDX_2:
        if types[ij] == "link":
            model.constraint_CS_capacity.add(model.pXi[ij] == 0)
        else:
            id = type_IDs[ij]

            extract_pois_0 = range(0, len(pois_0[pois_0[col_type_ID] == id]))
            extract_pois_1 = range(0, len(pois_1[pois_1[col_type_ID] == id]))

            if len(extract_pois_0) > 0 and len(extract_pois_1) > 0:
                ind_0 = extract_pois_0[0]
                ind_1 = extract_pois_1[0]
                model.constraint_CS_capacity.add(
                    model.pYi_dir_0[model.IDX_0[ind_0]]
                    + model.pYi_dir_1[model.IDX_1[ind_1]]
                    <= g * model.pXi[ij]
                )
                model.constraint_CS_capacity.add(
                    (
                        model.pYi_dir_0[model.IDX_0[ind_0]]
                        + model.pYi_dir_1[model.IDX_1[ind_1]]
                    )
                    * energy
                    >= sum(
                        [
                            model.pE_charged_0[kl, model.IDX_0[ind_0]]
                            for kl in model.IDX_0
                        ]
                    )
                    + sum(
                        [
                            model.pE_charged_1[kl, model.IDX_0[ind_0] + n1]
                            for kl in model.IDX_1
                        ]
                    )
                    + sum(
                        [
                            model.pE_charged_1[kl, model.IDX_1[ind_1]]
                            for kl in model.IDX_1
                        ]
                    )
                    + sum(
                        [
                            model.pE_charged_0[kl, model.IDX_1[ind_1] + n0]
                            for kl in model.IDX_0
                        ]
                    )
                )
            elif len(extract_pois_0) > 0:

                ind_0 = extract_pois_0[0]
                model.constraint_CS_capacity.add(
                    model.pYi_dir_0[model.IDX_0[ind_0]] <= g * model.pXi[ij]
                )
                model.constraint_CS_capacity.add(
                    model.pYi_dir_0[model.IDX_0[ind_0]] * energy
                    >= sum(
                        [
                            model.pE_charged_0[kl, model.IDX_0[ind_0]]
                            for kl in model.IDX_0
                        ]
                    )
                    + sum(
                        [
                            model.pE_charged_1[kl, model.IDX_0[ind_0] + n1]
                            for kl in model.IDX_1
                        ]
                    )
                )

            elif len(extract_pois_1) > 0:

                ind_1 = extract_pois_1[0]
                model.constraint_CS_capacity.add(
                    model.pYi_dir_1[model.IDX_1[ind_1]] <= g * model.pXi[ij]
                )
                model.constraint_CS_capacity.add(
                    model.pYi_dir_1[model.IDX_1[ind_1]] * energy
                    >= sum(
                        [
                            model.pE_charged_1[kl, model.IDX_1[ind_1]]
                            for kl in model.IDX_1
                        ]
                    )
                    + sum(
                        [
                            model.pE_charged_0[kl, model.IDX_1[ind_1] + n0]
                            for kl in model.IDX_0
                        ]
                    )
                )


def constraint_input_at_source_nodes(model):

    model.constraint_input_at_source_nodes = ConstraintList()

    for ij in model.IDX_0:
        model.constraint_input_at_source_nodes.add(model.pE_input_0[ij, ij] == 0)

    for ij in model.IDX_1:
        model.constraint_input_at_source_nodes.add(model.pE_input_1[ij, ij] == 0)


def constraint_input_output(model, pois_df_0, pois_df_1, links_gdf, segments_gdf):

    model.constraint_input_output = ConstraintList()
    segment_ids_0 = pois_df_0.segment_id.to_list()
    segment_ids_1 = pois_df_1.segment_id.to_list()

    poi_indices_0 = pois_df_0.index.to_list()
    poi_indices_1 = pois_df_1.index.to_list()
    direction = 0
    input_0 = np.zeros([n0, n0+n1])
    input_1 = np.zeros([n1, n0+n1])
    output_0 = np.zeros([n0, n0+n1])
    output_1 = np.zeros([n1, n0+n1])

    const_input_0 = np.zeros([n0, n0+n1])
    const_output_0 = np.zeros([n0, n0+n1])

    const_input_1 = np.zeros([n1, n0+n1])
    const_output_1 = np.zeros([n1, n0+n1])

    no_io_indices_0 = []
    no_io_indices_1 = []
    for ij in model.IDX_0:
        ind = poi_indices_0[ij]
        seg_id = segment_ids_0[ij]

        output, path = constraint_input_ouput_relations_for_pois_on_segm(
            seg_id,
            direction,
            model,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        io_rels = output[0]
        filtered_rels = [io[1] for io in io_rels if io[0] == (ind, direction)]

        ind_mat = poi_indices_0.index(ind)
        if len(filtered_rels) == 0:
            no_io_indices_0.append(ij)
        for rel in filtered_rels:
            prev_dir = rel[0][1]
            next_dir = rel[1][1]
            if prev_dir == 0:
                prev_ind = poi_indices_0.index(rel[0][0])
            else:
                prev_ind = poi_indices_1.index(rel[0][0])

            if next_dir == 0:
                next_ind = poi_indices_0.index(rel[1][0])
            else:
                next_ind = poi_indices_1.index(rel[1][0])

            if prev_dir == 0 and next_dir == 0:
                model.constraint_input_output.add(model.pE_output_0[ind_mat, prev_ind] == model.pE_input_0[ind_mat,
                                                                                                          next_ind])
                output_0[ind_mat, prev_ind] = 1
                input_0[ind_mat, next_ind] = 1
            elif prev_dir == 1 and next_dir == 1:
                model.constraint_input_output.add(model.pE_output_0[ind_mat, n0 + prev_ind] == model.pE_input_0[ind_mat,
                                                                                                           n0 + next_ind])
                output_0[ind_mat, n0 + prev_ind] = 1
                input_0[ind_mat, n0 + next_ind] = 1
    direction = 1
    for ij in model.IDX_1:
        ind = poi_indices_1[ij]
        seg_id = segment_ids_1[ij]

        output, path = constraint_input_ouput_relations_for_pois_on_segm(
            seg_id,
            direction,
            model,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        io_rels = output[0]
        filtered_rels = [io[1] for io in io_rels if io[0] == (ind, direction)]
        if len(filtered_rels) == 0:
            no_io_indices_1.append(ij)
        ind_mat = poi_indices_1.index(ind)
        for rel in filtered_rels:
            prev_dir = rel[0][1]
            next_dir = rel[1][1]
            if prev_dir == 0:
                prev_ind = poi_indices_0.index(rel[0][0])
            else:
                prev_ind = poi_indices_1.index(rel[0][0])

            if next_dir == 0:
                next_ind = poi_indices_0.index(rel[1][0])
            else:
                next_ind = poi_indices_1.index(rel[1][0])

            if prev_dir == 0 and next_dir == 0:
                model.constraint_input_output.add(model.pE_output_1[ind_mat, n1 + prev_ind] == model.pE_input_1[ind_mat,
                                                                                                           n1 + next_ind])
                output_0[ind_mat, n1 +prev_ind] = 1
                input_0[ind_mat, n1 + next_ind] = 1
            elif prev_dir == 1 and next_dir == 1:
                model.constraint_input_output.add(model.pE_output_1[ind_mat, prev_ind] == model.pE_input_1[ind_mat,
                                                                                                           next_ind])
                output_1[ind_mat, prev_ind] = 1
                input_1[ind_mat, next_ind] = 1

    for ij in model.IDX_0:
        if ij in no_io_indices_0:
            model.constraint_input_output.add(model.pE_input_0[ij, ij] == 0)
            model.constraint_input_output.add(model.pE_output_0[ij, ij] == 0)
            const_input_0[ij, ij] = 1
            const_output_0[ij, ij] = 1
        for kl in model.IDX_0:
            if input_0[ij, kl] == 0 and output_0[ij, kl] == 1:
                model.constraint_input_output.add(model.pE_input_0[ij, kl] == 0)
                const_input_0[ij, kl] = 1
            elif input_0[ij, kl] == 1 and output_0[ij, kl] == 0:
                model.constraint_input_output.add(model.pE_output_0[ij, kl] == 0)
                const_output_0[ij, kl] = 1

    for ij in model.IDX_1:
        if ij in no_io_indices_1:
            model.constraint_input_output.add(model.pE_input_1[ij, ij] == 0)
            model.constraint_input_output.add(model.pE_output_1[ij, ij] == 0)
            const_input_1[ij, ij] = 1
            const_output_1[ij, ij] = 1
        for kl in model.IDX_1:
            if input_1[ij, kl] == 0 and output_1[ij, kl] == 1:
                model.constraint_input_output.add(model.pE_input_1[ij, kl] == 0)
                const_input_1[ij, kl] = 1

            elif input_1[ij, kl] == 1 and output_1[ij, kl] == 0:
                model.constraint_input_output.add(model.pE_output_1[ij, kl] == 0)
                const_output_1[ij, kl] = 1


def constraint_output(model, pois_df_0, pois_df_1):
    # identify input nodes which don't have outputs
    # identify output nods which are not inputs

    model.constraint_output = ConstraintList()

    pois_indices_0 = pois_df_0.index.to_list()
    pois_indices_1 = pois_df_1.index.to_list()

    seg_ids_0 = pois_df_0.segment_id.to_list()
    seg_ids_1 = pois_df_1.segment_id.to_list()

    direction_0 = 0
    direction_1 = 1
    output_00 = np.zeros([n0, n0+n1])
    output_11 = np.zeros([n1, n0+n1])
    for kl in model.IDX_0:
        output, path = constraint_input_ouput_relations_for_pois_on_segm(
            seg_ids_0[kl],
            direction_0,

            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        ending_nodes = output[1]
        filter_end_node = [io for io in ending_nodes if io[0] == (pois_indices_0[kl], direction_0)]

        if len(filter_end_node) == 0:
            model.constraint_output.add(model.pE_output_0[kl, kl] == 0)
            output_00[kl, kl] = 1
        else:
            for f in filter_end_node:
                if f[1][3] == 0:
                    model.constraint_output.add(model.pE_output_0[kl, pois_indices_0.index(f[1][2])] == 0)
                    output_00[kl, pois_indices_0.index(f[1][2])] = 1
                else:
                    model.constraint_output.add(model.pE_output_0[kl, n0 + pois_indices_1.index(f[1][2])] == 0)
                    output_00[kl, n0 + pois_indices_1.index(f[1][2])] = 1

    for kl in model.IDX_1:
        output, path = constraint_input_ouput_relations_for_pois_on_segm(
            seg_ids_1[kl],
            direction_1,
            model,
            pois_df,
            pois_df_0,
            pois_df_1,
            links_gdf,
            segments_gdf,
            dmax,
        )
        ending_nodes = output[1]
        filter_end_node = [io for io in ending_nodes if io[0] == (pois_indices_1[kl], direction_1)]

        if len(filter_end_node) == 0:
            model.constraint_output.add(model.pE_output_1[kl, kl] == 0)
            output_11[kl, kl] = 1

        else:
            for f in filter_end_node:
                if f[1][3] == 0:
                    model.constraint_output.add(model.pE_output_1[kl, n1 + pois_indices_0.index(f[1][2])] == 0)
                    output_11[kl, n1 + pois_indices_0.index(f[1][2])] = 1
                else:
                    model.constraint_output.add(model.pE_output_1[kl, pois_indices_1.index(f[1][2])] == 0)
                    output_11[kl, pois_indices_1.index(f[1][2])] = 1


def constraint_zero(model):
    model.constraint_zero = ConstraintList()

    for ij in model.IDX_0:
        for kl in model.IDX_3:
            model.constraint_zero.add(model.pE_input_0[ij, kl] >= 0)
            model.constraint_zero.add(model.pE_charged_0[ij, kl] >= 0)
            model.constraint_zero.add(model.pE_output_0[ij, kl] >= 0)
            model.constraint_zero.add(model.test_var_0[ij, kl] >= 0)

    for ij in model.IDX_1:
        for kl in model.IDX_3:
            model.constraint_zero.add(model.pE_charged_1[ij, kl] >= 0)
            model.constraint_zero.add(model.pE_input_1[ij, kl] >= 0)
            model.constraint_zero.add(model.pE_output_1[ij, kl] >= 0)
            model.constraint_zero.add(model.test_var_1[ij, kl] >= 0)


def complement_rels(io_relations, pois_0, pois_1, links_gdf, path):
    io_relations = io_relations.copy()
    linkage_dir, link_nodes = equal_linkages_points(pois_0, pois_1, links_gdf)
    new_rels = []
    # set of source_nodes
    source_nodes = list(set([io[0] for io in io_relations]))
    # TODO: only add new IO-rel if seg_id of upcoming is a child of the other
    for ij in range(0, len(source_nodes)):
        sn = source_nodes[ij]
        filtered_rels = [io[1] for io in io_relations if io[0] == sn]
        out_rels = [io[0] for io in filtered_rels]
        in_rels = [io[1] for io in filtered_rels]
        for n in in_rels:
            if n[1] == 0:
                prev_seg = pois_0[pois_0.index==n[0]].segment_id.to_list()[0]
            else:
                prev_seg = pois_1[pois_1.index == n[0]].segment_id.to_list()[0]
            # finding key in path
            prev_seg_key = None
            for k in path.keys():
                if path[k].name == (prev_seg, n[1]):
                    prev_seg_key = k
                    break

            if is_linkage(n, pois_0, pois_1, links_gdf)[0]:
                k = is_linkage(n, pois_0, pois_1, links_gdf)[1]
                ln = linkage_dir[k]
                for mn in ln:
                    if mn[1] == 0:
                        next_seg = pois_0[pois_0.index == mn[0]].segment_id.to_list()[0]
                    else:
                        next_seg = pois_1[pois_1.index == mn[0]].segment_id.to_list()[0]

                    next_seg_key = None
                    for k in path.keys():
                        if path[k].name == (next_seg, mn[1]):
                            next_seg_key = k
                            break

                    if mn in out_rels and path[next_seg_key] in path[prev_seg_key].children:
                        new_rels.append((sn, (n, mn)))

        if is_linkage(sn, pois_0, pois_1, links_gdf)[0]:
            k = is_linkage(sn, pois_0, pois_1, links_gdf)[1]
            ln = linkage_dir[k]
            for mn in ln:
                if mn in out_rels and not mn == sn:
                    new_rels.append((sn, (sn, mn)))

    return io_relations + new_rels


def add_ratios_to_io_rels(filtered_ios, pois_0, pois_1):
    singular_io_rels = list(set(filtered_ios))
    extended_information_rels = []
    for s in singular_io_rels:
        same_o = [s[1]] + [io[1] for io in singular_io_rels if io[0] == s[0] and not io == s]
        # collect all traffic_flow numbers for each same_o
        segs = [get_segment(same_o[ij][0], same_o[ij][1], pois_0, pois_1) for ij in range(0, len(same_o))]
        trafficflow_numbers = [get_traffic_count(segs[ij], same_o[ij][1], pois_0, pois_1) for ij in range(0, len(segs))]

        #if len(same_o) > 1:
        extended_information_rels.append((s + (trafficflow_numbers[0]/sum(trafficflow_numbers),)))

    return extended_information_rels


def create_mask_enum(model, pois_df_0, pois_df_1, links_gdf, segments_gdf, dmax):

    model.constraint_io = ConstraintList()
    n0 = len(pois_df_0)
    n1 = len(pois_df_1)

    path_directory = {}

    segment_ids_0 = pois_df_0.segment_id.to_list()
    segment_ids_1 = pois_df_1.segment_id.to_list()

    poi_indices_0 = pois_df_0.index.to_list()
    poi_indices_1 = pois_df_1.index.to_list()
    direction = 0
    input_0 = np.zeros([n0, n0 + n1])
    input_1 = np.zeros([n1, n0 + n1])
    output_0 = np.zeros([n0, n0 + n1])
    output_1 = np.zeros([n1, n0 + n1])

    const_input_0 = np.zeros([n0, n0 + n1])
    const_output_0 = np.zeros([n0, n0 + n1])

    const_input_1 = np.zeros([n1, n0 + n1])
    const_output_1 = np.zeros([n1, n0 + n1])

    no_io_indices_0 = []
    no_io_indices_1 = []
    prev_seg = None
    for ij in range(0, n0):
        ind = poi_indices_0[ij]
        seg_id = segment_ids_0[ij]
        if not prev_seg == seg_id:
            output, path = constraint_input_ouput_relations_for_pois_on_segm(
                seg_id,
                direction,
                pois_df,
                pois_df_0,
                pois_df_1,
                links_gdf,
                segments_gdf,
                dmax,
            )

            path_directory[str(seg_id) + '_' + str(direction)] = path
            io_rels = output[0]
            io_rels = complement_rels(io_rels, pois_df_0, pois_df_1, links_gdf, path)

        filtered_rels = add_ratios_to_io_rels([io[1] for io in io_rels if io[0] == (ind, direction)], pois_df_0, pois_df_1)
        out_rels = [io[0] for io in filtered_rels]
        in_rels = [io[1] for io in filtered_rels]

        ind_mat = poi_indices_0.index(ind)
        if len(filtered_rels) == 0:
            no_io_indices_0.append(ij)
        for rel in filtered_rels:
            prev_dir = rel[0][1]
            next_dir = rel[1][1]
            if prev_dir == 0:
                prev_ind = poi_indices_0.index(rel[0][0])
            else:
                prev_ind = poi_indices_1.index(rel[0][0])

            if next_dir == 0:
                next_ind = poi_indices_0.index(rel[1][0])
            else:
                next_ind = poi_indices_1.index(rel[1][0])
            ratio = rel[2]
            if prev_dir == 0 and next_dir == 0:
                output_0[ind_mat, prev_ind] = 1
                input_0[ind_mat, next_ind] = 1
                model.constraint_io.add(model.pE_output_0[ind_mat, prev_ind] * ratio == model.pE_input_0[ind_mat, next_ind])

            elif prev_dir == 1 and next_dir == 1:
                output_0[ind_mat, n0 + prev_ind] = 1
                input_0[ind_mat, n0 + next_ind] = 1
                model.constraint_io.add(
                    model.pE_output_0[ind_mat, n0 + prev_ind] * ratio == model.pE_input_0[ind_mat, n0 + next_ind])
            elif prev_dir == 0 and next_dir == 1:
                output_0[ind_mat, prev_ind] = 1
                input_0[ind_mat, n0 + next_ind] = 1
                model.constraint_io.add(
                    model.pE_output_0[ind_mat, prev_ind] * ratio == model.pE_input_0[ind_mat, n0 + next_ind])
            else:
                output_0[ind_mat, n0 + prev_ind] = 1
                input_0[ind_mat, next_ind] = 1
                model.constraint_io.add(
                    model.pE_output_0[ind_mat, n0 + prev_ind] * ratio == model.pE_input_0[ind_mat, next_ind])
        prev_seg = seg_id

    prev_seg = None
    direction = 1
    for ij in range(0, n1):
        ind = poi_indices_1[ij]
        seg_id = segment_ids_1[ij]

        if not prev_seg == seg_id:
            output, path = constraint_input_ouput_relations_for_pois_on_segm(
                seg_id,
                direction,
                pois_df,
                pois_df_0,
                pois_df_1,
                links_gdf,
                segments_gdf,
                dmax,
            )
            path_directory[str(seg_id) + '_' + str(direction)] = path
            io_rels = output[0]
            io_rels = complement_rels(io_rels, pois_df_0, pois_df_1, links_gdf, path)
        filtered_rels = add_ratios_to_io_rels([io[1] for io in io_rels if io[0] == (ind, direction)], pois_df_0, pois_df_1)
        out_rels = [io[0] for io in filtered_rels]
        in_rels = [io[1] for io in filtered_rels]
        if len(filtered_rels) == 0:
            no_io_indices_1.append(ij)
        ind_mat = poi_indices_1.index(ind)
        for rel in filtered_rels:
            prev_dir = rel[0][1]
            next_dir = rel[1][1]
            if prev_dir == 0:
                prev_ind = poi_indices_0.index(rel[0][0])
            else:
                prev_ind = poi_indices_1.index(rel[0][0])

            if next_dir == 0:
                next_ind = poi_indices_0.index(rel[1][0])
            else:
                next_ind = poi_indices_1.index(rel[1][0])
            ratio = rel[2]
            if prev_dir == 0 and next_dir == 0:
                model.constraint_io.add(
                    model.pE_output_1[ind_mat, n1 + prev_ind] * ratio == model.pE_input_1[ind_mat, n1 + next_ind])
                output_1[ind_mat, n1 + prev_ind] = 1
                input_1[ind_mat, n1 + next_ind] = 1

            elif prev_dir == 1 and next_dir == 1:
                output_1[ind_mat, prev_ind] = 1
                input_1[ind_mat, next_ind] = 1
                model.constraint_io.add(model.pE_output_1[ind_mat, prev_ind] * ratio == model.pE_input_1[ind_mat, next_ind])
            elif prev_dir == 1 and next_dir == 0:
                output_1[ind_mat, prev_ind] = 1
                input_1[ind_mat, n1 + next_ind] = 1
                model.constraint_io.add(model.pE_output_1[ind_mat, prev_ind] * ratio == model.pE_input_1[ind_mat, n1 + next_ind])
            else:
                output_1[ind_mat, n1 + prev_ind] = 1
                input_1[ind_mat, next_ind] = 1
                model.constraint_io.add(model.pE_output_1[ind_mat, n1 + prev_ind] * ratio== model.pE_input_1[ind_mat, next_ind])
        prev_seg = seg_id

    for ij in range(0, n0):
        if ij in no_io_indices_0:
            const_output_0[ij, ij] = 1
        for kl in range(0, n0+n1):
            if input_0[ij, kl] == 0 and output_0[ij, kl] == 1:
                const_input_0[ij, kl] = 1
            elif input_0[ij, kl] == 1 and output_0[ij, kl] == 0:
                const_output_0[ij, kl] = 1

    for ij in range(0, n0):
        for kl in range(0, n0 + n1):
            const_input_0[ij, ij] = 1

    for ij in range(0, n1):
        if ij in no_io_indices_1:
            const_input_1[ij, ij] = 1
            const_output_1[ij, ij] = 1
        for kl in range(0, n0+n1):
            if input_1[ij, kl] == 0 and output_1[ij, kl] == 1:
                const_input_1[ij, kl] = 1

            elif input_1[ij, kl] == 1 and output_1[ij, kl] == 0:
                const_output_1[ij, kl] = 1

    mask_0 = np.zeros([n0, n0 + n1])
    mask_1 = np.zeros([n1, n0 + n1])

    for ij in range(0, n0):
        for kl in range(0, n0 + n1):
            if input_0[ij, kl] == 1 or output_0[ij, kl] == 1 or const_input_0[ij, kl] == 1 or const_output_0[ij, kl] == 1:
                mask_0[ij, kl] = 1

    for ij in range(0, n1):
        for kl in range(0, n0 + n1):
            if input_1[ij, kl] == 1 or output_1[ij, kl] == 1 or const_input_1[ij, kl] == 1 or const_output_1[ij, kl] == 1:
                mask_1[ij, kl] = 1

    return const_input_0, const_input_1, const_output_0, const_output_1, mask_0, mask_1, path_directory


