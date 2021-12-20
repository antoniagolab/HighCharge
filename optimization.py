"""
Created by Antonia Golab
Optimization of charging station allocation
last edit: 2021/12/08
"""


# TODO: reformat and make modular !!
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


import pandas as pd
import numpy as np
from pyomo.environ import *
from optimization_parameters import *
from optimization_additional_functions import *
from integrating_exxisting_infrastructure import *
import time
from visualize_results import *
from termcolor import colored
from optimization_utils import *


def optimization(
    dmax=dmax,
    eta=eta,
    ec=ec,
    acc=acc,
    charging_capacity=charging_capacity,
    specific_demand=specific_demand,
    introduce_existing_infrastructure=introduce_existing_infrastructure
):
    """
    Constraint and objective function definition + solution of optimization using parameters defined in
    optimization_parameters.py
    :return:
    """
    if no_new_infrastructure:
        introduce_existing_infrastructure = True

    start = time.time()
    # ------------------------------------------ printing input parameters -------------------------------------------
    print("------------------------------------------")
    print("Input data:")
    print(
        "nb. of network segments: ",
        str(len(segments_gdf)),
        "\nnb. of nodes: ",
        str(len(pois_df)),
        "; type 1: ",
        str(len(pois_df[pois_df[col_type] == "ra"])),
        "; type 2: ",
        str(len(pois_df[pois_df[col_type] == "link"])),
    )
    print("Input parameters:")
    print(
        "specific_demand=",
        str(specific_demand),
        "; cx=",
        str(cfix),
        "; cy=",
        str(cvar),
        "; eta=",
        str(eta),
        "; mu=",
        str(mu),
        "; gamma_h=",
        str(gamma_h),
        "; dist_max=",
        dmax,
        "; introduce_existing_infrastructure=",
        str(introduce_existing_infrastructure),
        "; no_new_infrastructure=",
        str(no_new_infrastructure)
    )
    print("------------------------------------------")

    # --------------------------------------------- model initialization ---------------------------------------------

    print("Model initialization ...")
    model = ConcreteModel()

    # variable definition
    # driving directions "normal" (=0) and "inverse" (=1) are treated separately throughout the optimization model
    # charging demand at each resting area is then summed up if it is for both directions

    model.IDX_0 = range(n0)
    model.IDX_1 = range(n1)
    model.IDX_2 = range(0, n3)
    model.IDX_3 = range(n0 + n1)

    # binary variables stating whether a charging station is installed at given ra (=resting area)
    model.pXi = Var(model.IDX_2, within=Binary)

    # integer variables indicating number of charging poles at a charging station
    model.pYi_dir_0 = Var(model.IDX_0, within=Integers)
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    # a fictitious energy load is passed on from ra to ra along a highway which (pE_Zu) which is the difference between
    # energy demand and covered energy demand at a resting area (pE_Laden); the remaining energy is passed on to the
    # next ra (pE_Ab)
    model.pE_input_0 = Var(model.IDX_0, model.IDX_3)
    model.pE_input_1 = Var(model.IDX_1, model.IDX_3)
    model.pE_output_0 = Var(model.IDX_0, model.IDX_3)
    model.pE_output_1 = Var(model.IDX_1, model.IDX_3)
    model.pE_charged_0 = Var(model.IDX_0, model.IDX_3)
    model.pE_charged_1 = Var(model.IDX_1, model.IDX_3)

    model.test_var_0 = Var(model.IDX_0, model.IDX_3)
    model.test_var_1 = Var(model.IDX_0, model.IDX_3)

    energy = charging_capacity  # (kWh) charging energy per day by one charging pole


    # ------------------------------------------------- constraints -------------------------------------------------
    print("------------------------------------------")
    print("Adding constraints ...")
    # the pE_Zu needs to be 0 at each first ra as it is assumed that no energy demand is previously covered;
    # all energy demand is constrained to be covered at the last ra on a highway

    print("\nAdding constraint: Input-Output-relations ...")
    t0 = time.time()
    (
        const_input_0,
        const_input_1,
        const_output_0,
        const_output_1,
        mask_0,
        mask_1,
        path_directory,
    ) = create_mask_enum(model, dir_0, dir_1, links_gdf, segments_gdf, dmax)
    print("... took ", str(time.time() - t0), " sec")

    print(
        "Adding constraint: Avoiding installation of charging stations at intersections ..."
    )
    t1 = time.time()
    constraint_Y_i(model, dir_0, dir_1)
    print("... took ", str(time.time() - t1), " sec")

    print(
        "Adding constraint: Inserting limitations to energy shifting, enforcing maximum possible energy coverage ..."
    )
    t2 = time.time()
    not_charged_energy = 0
    model.c1 = ConstraintList()
    # direction == 0
    for ij in model.IDX_0:
        seg_id = dir_0[dir_0.index == ij].segment_id.to_list()[0]
        path = path_directory[str(seg_id) + "_0"]
        for kl in model.IDX_3:
            if mask_0[ij, kl] == 0:
                model.c1.add(model.pE_input_0[ij, kl] == 0)
                model.c1.add(model.pE_output_0[ij, kl] == 0)
                model.c1.add(model.pE_charged_0[ij, kl] == 0)
            if const_input_0[ij, kl] == 1:
                model.c1.add(model.pE_input_0[ij, kl] == 0)
            if const_output_0[ij, kl] == 1:
                if kl < n0:
                    end_seg_id = dir_0[dir_0.index == kl].segment_id.to_list()[0]
                    end_direction = 0
                else:
                    end_seg_id = dir_1[dir_1.index == kl - n0].segment_id.to_list()[0]
                    end_direction = 1
                if not are_ras_along_the_way(
                    ij,
                    seg_id,
                    0,
                    path,
                    dir_0,
                    dir_1,
                    kl,
                    end_seg_id,
                    end_direction,
                    n0,
                    n1,
                    segments_gdf,
                )[0]:
                    factor = are_ras_along_the_way(
                        ij,
                        seg_id,
                        0,
                        path,
                        dir_0,
                        dir_1,
                        kl,
                        end_seg_id,
                        end_direction,
                        n0,
                        n1,
                        segments_gdf,
                    )[1]
                    print(ij, kl, energy_demand_matrix_0[ij, ij] * factor, factor)
                    model.c1.add(
                        model.pE_output_0[ij, kl]
                        <= energy_demand_matrix_0[ij, ij] * factor
                    )
                    not_charged_energy = (
                        not_charged_energy + energy_demand_matrix_0[ij, ij] * factor
                    )
                else:
                    model.c1.add(model.pE_output_0[ij, kl] == 0)

    # direction == 1
    for ij in model.IDX_1:
        seg_id = dir_1[dir_1.index == ij].segment_id.to_list()[0]
        path = path_directory[str(seg_id) + "_1"]
        for kl in model.IDX_3:
            if mask_1[ij, kl] == 0:
                model.c1.add(model.pE_input_1[ij, kl] == 0)
                model.c1.add(model.pE_output_1[ij, kl] == 0)
                model.c1.add(model.pE_charged_1[ij, kl] == 0)
            if const_input_1[ij, kl] == 1:
                model.c1.add(model.pE_input_1[ij, kl] == 0)

            if const_output_1[ij, kl] == 1:
                if kl < n1:
                    end_seg_id = dir_1[dir_1.index == kl].segment_id.to_list()[0]
                    end_direction = 1
                else:
                    end_seg_id = dir_0[dir_0.index == kl - n1].segment_id.to_list()[0]
                    end_direction = 0
                if not are_ras_along_the_way(
                    ij,
                    seg_id,
                    1,
                    path,
                    dir_0,
                    dir_1,
                    kl,
                    end_seg_id,
                    end_direction,
                    n0,
                    n1,
                    segments_gdf,
                )[0]:
                    factor = are_ras_along_the_way(
                        ij,
                        seg_id,
                        1,
                        path,
                        dir_0,
                        dir_1,
                        kl,
                        end_seg_id,
                        end_direction,
                        n0,
                        n1,
                        segments_gdf,
                    )[1]
                    print(ij, kl, energy_demand_matrix_0[ij, ij] * factor, factor)
                    model.c1.add(
                        model.pE_output_1[ij, kl]
                        <= energy_demand_matrix_1[ij, ij] * factor
                    )
                    not_charged_energy = (
                        not_charged_energy + energy_demand_matrix_0[ij, ij] * factor
                    )
                else:
                    model.c1.add(model.pE_output_1[ij, kl] == 0)
    print("... took ", str(time.time() - t2), " sec")

    print("Adding constraint: net balance energy at each node ...")
    t3 = time.time()
    constraint_rel_dem_i_o_charge(
        model, energy_demand_matrix_0, energy_demand_matrix_1, dir_0, dir_1, links_gdf
    )
    print("... took ", str(time.time() - t3), " sec")

    print(
        "Adding constraints: existing infrastructure, defining energy demand coverage by charging pole, ..."
    )
    t4 = time.time()
    model.c_profitable_stations = ConstraintList()
    model.c12 = ConstraintList()
    model.installed_infrastructure = ConstraintList()
    dir_names = dir[col_POI_ID].to_list()
    dir_directions = dir[col_directions].to_list()
    dir_highways = dir[col_segment_id].to_list()
    dir_type_ids = dir[col_type_ID].to_list()
    installed_stations = 0
    installed_poles = 0

    for ij in model.IDX_2:
        name = dir_names[ij]
        direc = dir_directions[ij]
        highway = dir_highways[ij]
        type_id = dir_type_ids[ij]
        extract_dir_0 = dir_0[
            (
                (dir_0[col_POI_ID] == name)
                & (dir_0[col_directions] == direc)
                & (dir_0[col_segment_id] == highway)
            )
        ].index.to_list()
        extract_dir_1 = dir_1[
            (
                (dir_1[col_POI_ID] == name)
                & (dir_1[col_directions] == direc)
                & (dir_1[col_segment_id] == highway)
            )
        ].index.to_list()
        if introduce_existing_infrastructure:
            ex_infr_0 = existing_infr_0[
                (
                    (existing_infr_0[col_type_ID] == type_id)
                    & (existing_infr_0[col_directions] == direc)
                    & (existing_infr_0[col_segment_id] == highway)
                    & (existing_infr_0["cs_below_50kwh"] > 0)
                )
            ].index.to_list()

            ex_infr_1 = existing_infr_1[
                (
                    (existing_infr_1[col_type_ID] == type_id)
                    & (existing_infr_1[col_directions] == direc)
                    & (existing_infr_1[col_segment_id] == highway)
                    & (existing_infr_1["cs_below_50kwh"] > 0)
                )
            ].index.to_list()

        if len(extract_dir_0) > 0 and len(extract_dir_1) > 0:
            ind_0 = extract_dir_0[0]
            ind_1 = extract_dir_1[0]
            model.c12.add(
                model.pYi_dir_0[model.IDX_0[ind_0]]
                + model.pYi_dir_1[model.IDX_1[ind_1]]
                <= g * model.pXi[ij]
            )
            model.c12.add(
                (
                    model.pYi_dir_0[model.IDX_0[ind_0]]
                    + model.pYi_dir_1[model.IDX_1[ind_1]]
                )
                * energy
                >= sum(
                    [model.pE_charged_0[kl, model.IDX_0[ind_0]] for kl in model.IDX_0]
                )
                + sum(
                    [model.pE_charged_1[kl, model.IDX_1[ind_1]] for kl in model.IDX_1]
                )
                + sum(
                    [
                        model.pE_charged_1[kl, n1 + model.IDX_0[ind_0]]
                        for kl in model.IDX_1
                    ]
                )
                + sum(
                    [
                        model.pE_charged_0[kl, n0 + model.IDX_1[ind_1]]
                        for kl in model.IDX_0
                    ]
                )
            )
            if introduce_existing_infrastructure:
                if len(ex_infr_0) > 0 and len(ex_infr_1) > 0:
                    infr_0 = ex_infr_0[0]
                    infr_1 = ex_infr_1[0]
                    model.c12.add(
                        model.pYi_dir_0[model.IDX_0[ind_0]]
                        + model.pYi_dir_1[model.IDX_1[ind_1]]
                        >= max(
                            existing_infr_0.at[infr_0, "cs_below_50kwh"],
                            existing_infr_1.at[infr_1, "cs_below_50kwh"],
                        )
                    )
                    model.c12.add(model.pXi[ij] == 1)
                    installed_stations = installed_stations + 1
                    installed_poles = (
                        max(
                            existing_infr_0.at[infr_0, "cs_below_50kwh"],
                            existing_infr_1.at[infr_1, "cs_below_50kwh"],
                        )
                        + installed_poles
                    )
                    if no_new_infrastructure:
                        model.c12.add(
                            model.pYi_dir_0[model.IDX_0[ind_0]]
                            + model.pYi_dir_1[model.IDX_1[ind_1]]
                            == max(
                                existing_infr_0.at[infr_0, "cs_below_50kwh"],
                                existing_infr_1.at[infr_1, "cs_below_50kwh"],
                            )
                        )
                elif no_new_infrastructure:
                    model.c12.add(
                        model.pYi_dir_0[model.IDX_0[ind_0]]
                        + model.pYi_dir_1[model.IDX_1[ind_1]]
                        == 0
                    )

        elif len(extract_dir_0) > 0:

            ind_0 = extract_dir_0[0]
            model.c12.add(model.pYi_dir_0[model.IDX_0[ind_0]] <= g * model.pXi[ij])
            model.c12.add(
                model.pYi_dir_0[model.IDX_0[ind_0]] * energy
                >= sum(
                    [model.pE_charged_0[kl, model.IDX_0[ind_0]] for kl in model.IDX_0]
                )
                + sum(
                    [
                        model.pE_charged_1[kl, n1 + model.IDX_0[ind_0]]
                        for kl in model.IDX_1
                    ]
                )
            )
            if introduce_existing_infrastructure:
                if len(ex_infr_0) > 0:
                    infr_0 = ex_infr_0[0]

                    model.c12.add(
                        model.pYi_dir_0[model.IDX_0[ind_0]]
                        >= existing_infr_0.at[infr_0, "cs_below_50kwh"]
                    )
                    model.c12.add(model.pXi[ij] == 1)
                    installed_stations = installed_stations + 1
                    installed_poles = (
                        existing_infr_0.at[infr_0, "cs_below_50kwh"] + installed_poles
                    )
                    if no_new_infrastructure:
                        model.c12.add(
                            model.pYi_dir_0[model.IDX_0[ind_0]]
                            == existing_infr_0.at[infr_0, "cs_below_50kwh"]
                        )
                elif no_new_infrastructure:
                    model.c12.add(
                        model.pYi_dir_0[model.IDX_0[ind_0]]
                        == 0
                    )

        elif len(extract_dir_1) > 0:

            ind_1 = extract_dir_1[0]
            model.c12.add(model.pYi_dir_1[model.IDX_1[ind_1]] <= g * model.pXi[ij])
            model.c12.add(
                model.pYi_dir_1[model.IDX_1[ind_1]] * energy
                >= sum(
                    [model.pE_charged_1[kl, model.IDX_1[ind_1]] for kl in model.IDX_1]
                )
                + sum(
                    [
                        model.pE_charged_0[kl, n0 + model.IDX_1[ind_1]]
                        for kl in model.IDX_0
                    ]
                )
            )
            if introduce_existing_infrastructure:
                if len(ex_infr_1) > 0:
                    infr_1 = ex_infr_1[0]
                    model.c12.add(
                        model.pYi_dir_1[model.IDX_1[ind_1]]
                        >= existing_infr_1.at[infr_1, "cs_below_50kwh"]
                    )
                    model.c12.add(model.pXi[ij] == 1)
                    installed_stations = installed_stations + 1
                    installed_poles = (
                        existing_infr_1.at[infr_1, "cs_below_50kwh"] + installed_poles
                    )
                    if no_new_infrastructure:
                        model.c12.add(
                            model.pYi_dir_1[model.IDX_1[ind_1]]
                            == existing_infr_1.at[infr_1, "cs_below_50kwh"]
                        )
                elif no_new_infrastructure:
                    model.c12.add(
                        model.pYi_dir_1[model.IDX_1[ind_1]]
                        == 0
                    )

    print("... took ", str(time.time() - t4), " sec")

    # zero-constraints
    print("Adding constraints: Zero constraints ...")
    t5 = time.time()
    constraint_zero(model)
    print("... took ", str(time.time() - t5), " sec")
    print("------------------------------------------")

    # ------------------------------------------------- objective -------------------------------------------------
    # minimization of installation costs
    print("Adding objective function ...")
    model.obj = Objective(
        expr=(
            (
                (
                    (sum(model.pXi[n] for n in model.IDX_2) - installed_stations) * cfix
                    + (
                        sum(model.pYi_dir_0[n] for n in model.IDX_0)
                        + sum(model.pYi_dir_1[n] for n in model.IDX_1)
                        - installed_poles
                    )
                    * cvar
                )
                + sum(model.test_var_0[m, n] for n in model.IDX_3 for m in model.IDX_0)
                * c_non_covered_demand
                + sum(model.test_var_1[m, n] for n in model.IDX_3 for m in model.IDX_1)
                * c_non_covered_demand
            )
        ),
        sense=minimize,
    )

    # ------------------------------------------------- solution --------------------------------------------------
    print("------------------------------------------")
    print("Model solution ...")
    t6 = time.time()
    opt = SolverFactory("gurobi")
    opt_success = opt.solve(model)
    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")
    print("... model solved in ", str(time.time() - t6), " sec")
    # -------------------------------------------------- results --------------------------------------------------

    pYi_dir_0 = np.array([model.pYi_dir_0[n].value for n in model.IDX_0])

    pYi_dir_1 = np.array([model.pYi_dir_1[n].value for n in model.IDX_1])
    pXi = np.array([model.pXi[n].value for n in model.IDX_2])
    pXi_dir_0 = np.where(pYi_dir_0 > 0, 1, 0)
    pXi_dir_1 = np.where(pYi_dir_1 > 0, 1, 0)
    pE_input_0_m = []
    pE_input_1_m = []
    pE_output_0_m = []
    pE_output_1_m = []
    pE_charged_0_m = []
    pE_charged_1_m = []
    test_var_0 = []
    test_var_1 = []

    for ij in model.IDX_0:
        pE_charged_0_m.append([model.pE_charged_0[ij, kl].value for kl in model.IDX_3])
        pE_input_0_m.append([model.pE_input_0[ij, kl].value for kl in model.IDX_3])
        pE_output_0_m.append([model.pE_output_0[ij, kl].value for kl in model.IDX_3])
        test_var_0.append([model.test_var_0[ij, kl].value for kl in model.IDX_3])

    for ij in model.IDX_1:
        pE_input_1_m.append([model.pE_input_1[ij, kl].value for kl in model.IDX_3])
        pE_charged_1_m.append([model.pE_charged_1[ij, kl].value for kl in model.IDX_3])
        pE_output_1_m.append([model.pE_output_1[ij, kl].value for kl in model.IDX_3])
        test_var_1.append([model.test_var_1[ij, kl].value for kl in model.IDX_3])

    pE_charged_0_m = np.array(pE_charged_0_m)
    pE_input_0_m = np.array(pE_input_0_m)
    pE_output_0_m = np.array(pE_output_0_m)
    pE_output_1_m = np.array(pE_output_1_m)
    pE_input_1_m = np.array(pE_input_1_m)
    pE_charged_1_m = np.array(pE_charged_1_m)
    test_var_0_m = np.array(test_var_0)
    test_var_1_m = np.array(test_var_1)

    pE_charged_0 = np.sum(pE_charged_0_m, axis=0)
    pE_charged_1 = np.sum(pE_charged_1_m, axis=0)
    pE_input_0 = np.sum(pE_input_0_m, axis=0)
    pE_input_1 = np.sum(pE_input_1_m, axis=0)
    pE_output_0 = np.sum(pE_output_0_m, axis=0)
    pE_output_1 = np.sum(pE_output_1_m, axis=0)

    objective_value = model.obj.value()
    installation_costs = (
        objective_value
        - (sum(sum(test_var_0_m)) + sum(sum(test_var_1_m))) * c_non_covered_demand
    )
    not_charged_2 = sum(sum(test_var_0_m)) + sum(sum(test_var_1_m))
    print("------------------------------------------")
    print(
        colored(
            "Total Profit installation costs: € " + str(installation_costs),
            "green",
        )
    )
    print(
        colored(
            "Total amount of charging poles: " + str(sum(pYi_dir_0) + sum(pYi_dir_1)),
            "green",
        ),
        colored("(existing: " + str(installed_poles) + ")", "red"),
    )
    print(
        colored("Total amount of charging stations: " + str(sum(pXi)), "green"),
        colored(" (existing: " + str(installed_stations) + ")", "red"),
    )
    print(
        colored(
            "Total energy charged: " + str(sum(pE_charged_0) + sum(pE_charged_1)),
            "green",
        ),
        colored(
            " (not charged: "
            + str(not_charged_energy)
            + ", "
            + str(not_charged_2)
            + ", "
            + str(
                (not_charged_energy + not_charged_2) / (sum(sum(energy_demand_matrix_0))
                + sum(sum(energy_demand_matrix_1)))
            )
            + "%)",
            "red",
        ),
    )
    print("------------------------------------------")

    # creating output file
    # merging both dataframes to singular table with all resting areas and calculated data
    output_cols = [
        col_segment_id,
        col_directions,
        col_type_ID,
        col_POI_ID,
        col_traffic_flow,
        col_energy_demand,
        col_distance,
        col_type,
    ]
    singular_info = [
        col_segment_id,
        col_directions,
        col_type_ID,
        col_POI_ID,
        col_distance,
        col_type,
    ]
    output_dir_0 = dir_0[output_cols]
    output_dir_1 = dir_1[output_cols]
    output_dir_0["pXi_dir"] = pXi_dir_0
    output_dir_1["pXi_dir"] = pXi_dir_1
    output_dir_0["pYi_dir_0"] = pYi_dir_0
    output_dir_1["pYi_dir_1"] = pYi_dir_1
    output_dir_0["pE_charged_0"] = np.sum(
        np.concatenate(
            (pE_charged_0_m[0:n0, 0:n0], pE_charged_1_m[0:n1, n1 : n1 + n0]), axis=0
        ),
        axis=0,
    )
    output_dir_1["pE_charged_1"] = np.sum(
        np.concatenate(
            (pE_charged_1_m[0:n1, 0:n1], pE_charged_0_m[0:n0, n0 : n0 + n1]), axis=0
        ),
        axis=0,
    )
    output_dir_0["pE_input_0"] = np.sum(
        np.concatenate(
            (pE_input_0_m[0:n0, 0:n0], pE_input_1_m[0:n1, n1 : n1 + n0]), axis=0
        ),
        axis=0,
    )
    output_dir_1["pE_input_1"] = np.sum(
        np.concatenate(
            (pE_input_1_m[0:n1, 0:n1], pE_input_0_m[0:n0, n0 : n0 + n1]), axis=0
        ),
        axis=0,
    )
    output_dir_0["pE_output_0"] = np.sum(
        np.concatenate(
            (pE_output_0_m[0:n0, 0:n0], pE_output_1_m[0:n1, n1 : n1 + n0]), axis=0
        ),
        axis=0,
    )
    output_dir_1["pE_output_1"] = np.sum(
        np.concatenate(
            (pE_output_1_m[0:n1, 0:n1], pE_output_0_m[0:n0, n0 : n0 + n1]), axis=0
        ),
        axis=0,
    )
    model_variable_names = ["pYi_dir", "pE_charged", "pE_input", "pE_output"]

    all_info_df = output_dir_0.append(output_dir_1)
    all_info_df.index = range(0, len(all_info_df))
    all_info_df = all_info_df.fillna(0.0)
    output_dataframe = pd.DataFrame()
    dir_0_names = dir_0[col_POI_ID].to_list()
    dir_1_names = dir_1[col_POI_ID].to_list()
    all_ra_names = list(set(dir_0_names + dir_1_names))
    output_dataframe.index = all_ra_names
    key_list = all_info_df.keys().to_list()
    for name in all_ra_names:
        extract_name_df = all_info_df[all_info_df[col_POI_ID] == name]
        for k in key_list:
            if k in singular_info:
                output_dataframe.loc[name, k] = extract_name_df[k].to_list()[0]
            elif k in output_cols:
                output_dataframe.loc[name, k] = extract_name_df[k].sum()
                if len(extract_name_df) > 1:
                    output_dataframe.loc[name, k + "_0"] = extract_name_df[k].to_list()[
                        0
                    ]
                    output_dataframe.loc[name, k + "_1"] = extract_name_df[k].to_list()[
                        1
                    ]
                elif name in dir_0_names:
                    output_dataframe.loc[name, k + "_0"] = extract_name_df[k].to_list()[
                        0
                    ]
                else:
                    output_dataframe.loc[name, k + "_1"] = extract_name_df[k].to_list()[
                        0
                    ]
        for k in model_variable_names:
            output_dataframe.loc[name, k] = (
                extract_name_df[k + "_0"].sum() + extract_name_df[k + "_1"].sum()
            )
            if len(extract_name_df) > 1:
                output_dataframe.loc[name, k + "_0"] = extract_name_df[
                    k + "_0"
                ].to_list()[0]
                output_dataframe.loc[name, k + "_1"] = extract_name_df[
                    k + "_1"
                ].to_list()[1]
            elif name in dir_0_names:
                output_dataframe.loc[name, k + "_0"] = extract_name_df[
                    k + "_0"
                ].to_list()[0]
            else:
                output_dataframe.loc[name, k + "_1"] = extract_name_df[
                    k + "_1"
                ].to_list()[0]

    output_dataframe["pXi"] = np.where((output_dataframe.pYi_dir > 0), 1, 0)
    output_dataframe = output_dataframe.fillna(0.0)
    output_filename = (
        "results/" + time_of_optimization + "_optimization_result_charging_stations.csv"
    )
    output_dataframe = output_dataframe.sort_values(by=[col_segment_id, col_distance])
    output_dataframe.to_csv(output_filename, index=False)

    # adding information on optimization
    additional_info = (
        "Time of calculation: "
        + time_of_optimization
        + "; "
        + "value of objective: "
        + str(model.obj.value())
        + ";  acc="
        + str(acc)
        + ";  ec="
        + str(ec)
        + ";  e_tax="
        + str(e_tax)
        + ";  cfix="
        + str(cfix)
        + ";  cvar="
        + str(cvar)
        + ";  eta="
        + str(eta)
        + ";  mu="
        + str(mu)
        + ";  i="
        + "\n"
    )

    file_opt_data = open(output_filename, "r")
    data = file_opt_data.read()
    file_opt_data.close()
    with open(output_filename[:-4] + "_all_info.txt", "w") as f:
        f.write(additional_info)
        f.write(data)
        f.close()

    visualize_results(
        filename=output_filename[:-4] + "_all_info.txt",
        maximum_dist_between_charging_stations=maximum_dist_between_charging_stations,
        eta=eta,
        ec=ec,
        acc=acc,
        charging_capacity=charging_capacity,
        energy=energy,
        specific_demand=specific_demand,
    )
    print("Total amount of computation: ", str(time.time() - start), " sec")

    return sum(pXi), sum(pYi_dir_0) + sum(pYi_dir_1), model.obj.value() * 365


if __name__ == "__main__":
    optimization(introduce_existing_infrastructure=introduce_existing_infrastructure)