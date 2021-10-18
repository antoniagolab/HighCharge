"""

Created by Antonia Golab

Optimization of charging station allocation

last edit: 2021/12/08

"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from optimization_parameters import *
from optimization_additional_functions import *
from integrating_exxisting_infrastructure import *
import time
import warnings
from pandas.core.common import SettingWithCopyWarning
from visualize_results import *
from termcolor import colored

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def optimization(
    maximum_dist_between_charging_stations=maximum_dist_between_charging_stations,
    eta=eta,
    ec=ec,
    acc=acc,
    charging_capacity=charging_capacity,
    specific_demand=specific_demand, input_existing_infrastructure=False
):
    """
    Constraint and objective function definition + solution of optimization using parameters defined in
    optimization_parameters.py

    :return:
    """
    model = ConcreteModel()  # Model initialization

    # --------------------------------------------- variable definition ---------------------------------------------
    # driving directions "normal" (=0) and "inverse" (=1) are treated separately throughout the optimization model
    # charging demand at each resting area is then summed up if it is for both directions

    model.IDX_0 = range(n0)
    model.IDX_1 = range(n1)
    model.IDX_2 = range(0, n3)

    # binary variables stating whether a charging station is installed at given ra (=resting area)
    # model.pXi_dir_0 = Var(model.IDX_0, within=Binary)
    # model.pXi_dir_1 = Var(model.IDX_1, within=Binary)
    model.pXi = Var(model.IDX_2, within=Binary)

    # integer variables indicating number of charging poles at a charging station
    model.pYi_dir_0 = Var(
        model.IDX_0,
        within=Integers,
    )
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    # a fictitious energy load is passed on from ra to ra along a highway which (pE_Zu) which is the difference between
    # energy demand and covered energy demand at a resting area (pE_Laden); the remaining energy is passed on to the
    # next ra (pE_Ab)
    model.pE_input_0 = Var(model.IDX_0)
    model.pE_input_1 = Var(model.IDX_1)
    model.pE_output_0 = Var(model.IDX_0)
    model.pE_output_1 = Var(model.IDX_1)
    model.pE_charged_0 = Var(model.IDX_0)
    model.pE_charged_1 = Var(model.IDX_1)

    cars_per_day = hours_of_constant_charging / (acc / charging_capacity)
    energy = acc * cars_per_day  # (kWh) charging energy per day by one charging pole
    # ------------------------------------------------- constraints -------------------------------------------------

    # the pE_Zu needs to be 0 at each first ra as it is assumed that no energy demand is previously covered;
    # all energy demand is constrained to be covered at the last ra on a highway
    model.c1 = ConstraintList()

    firsts_0 = dir_0[dir_0["first"] == True].index
    for ind in firsts_0:
        model.c1.add(model.pE_input_0[ind] == 0)
    lasts_0 = dir_0[dir_0["last"] == True].index
    for ind in lasts_0:
        model.c1.add(model.pE_output_0[ind] == 0)

    firsts_1 = dir_1[dir_1["first"] == True].index
    for ind in firsts_1:
        model.c1.add(model.pE_input_1[ind] == 0)
    lasts_1 = dir_1[dir_1["last"] == True].index
    for ind in lasts_1:
        model.c1.add(model.pE_output_1[ind] == 0)

    #
    model.c55 = ConstraintList()
    model.c55.add(model.pE_charged_0[model.IDX_0[21]] + model.pE_charged_0[model.IDX_0[22]]>= 1)
    model.c55.add(sum([model.pE_charged_0[ij] for ij in model.IDX_0]) >= 1)

    # constraints defining the passing on of the fictitious energy
    #
    # TODO: rethink here the direction!! dir_1 goes in a different direction
    #   -> also: this relationship does only stand within one highway -> this is not caught by this constraint
    model.c2 = ConstraintList()
    # for ij in range(1, n0):
    #     model.c2.add(model.pE_input_0[ij] == model.pE_output_0[ij - 1])
    #
    # for ij in range(1, n1):
    #     model.c2.add(model.pE_input_1[ij] == model.pE_output_1[ij - 1])

    for name in highway_names:
        if name in l0:
            dir0_extract_indices = dir_0[dir_0[col_highway] == name].index
            if len(dir0_extract_indices) > 0:
                for kl in range(1, len(dir0_extract_indices)):
                    model.c2.add(
                        model.pE_input_0[dir0_extract_indices[kl]] == model.pE_output_0[dir0_extract_indices[kl - 1]])

        if name in l1:
            dir1_extract_indices = dir_1[dir_1[col_highway] == name].index
            if len(dir1_extract_indices) > 0:
                for kl in range(len(dir1_extract_indices) - 2, -1, -1):
                    model.c2.add(
                        model.pE_input_1[dir1_extract_indices[kl]] == model.pE_output_1[dir1_extract_indices[kl] + 1])

                                                                                        # constraining the fictitiously passed on energy to ensure evenly distributed charging network
    #
    # dir_0_highways = dir_0[col_highway].to_list()
    # model.c3 = ConstraintList()
    # for ij in model.IDX_0:
    #     highway = dir_0_highways[ij]
    #     model.c3.add(
    #         model.pE_input_0[ij] / (traffic_flows_dir_0[ij] * specific_demand * eta)
    #         <= maximum_dist_between_charging_stations
    #     )
    #     model.c3.add(
    #         model.pE_output_0[ij] / (traffic_flows_dir_0[ij] * specific_demand * eta)
    #         <= maximum_dist_between_charging_stations
    #     )
    #
    # dir_1_highways = dir_1[col_highway].to_list()
    # model.c4 = ConstraintList()
    # for ij in model.IDX_1:
    #     highway = dir_1_highways[ij]
    #     model.c4.add(
    #         model.pE_input_1[ij] / (traffic_flows_dir_1[ij] * specific_demand * eta)
    #         <= maximum_dist_between_charging_stations
    #     )
    #     model.c4.add(
    #         model.pE_output_1[ij] / (traffic_flows_dir_1[ij] * specific_demand * eta)
    #         <= maximum_dist_between_charging_stations
    #     )

    dir_names = dir[col_rest_area_name].to_list()
    dir_directions = dir[col_directions].to_list()
    dir_highways = dir[col_highway].to_list()

    # defining the relationship between energy demand, demand coverage and net energy demand
    model.c7 = ConstraintList()
    for ij in model.IDX_0:
        model.c7.add(
            model.pE_charged_0[ij]
            - energy_demand_0[ij] * eta * mu
            - model.pE_input_0[ij]
            + model.pE_output_0[ij]
            == 0
        )

    model.c8 = ConstraintList()
    for ij in model.IDX_1:
        model.c8.add(
            model.pE_charged_1[ij]
            - energy_demand_1[ij] * eta * mu
            - model.pE_input_1[ij]
            + model.pE_output_1[ij]
            == 0
        )

    # defining how much energy demand a charging station is able to cover
    # TODO: SHARED usage -> one pole can be used by cars coming from both directions

    model.c9 = ConstraintList()
    for ij in model.IDX_0:
        model.c9.add(model.pYi_dir_0[ij] * energy >= model.pE_charged_0[ij])

    model.c10 = ConstraintList()
    for ij in model.IDX_1:
        model.c10.add(model.pYi_dir_1[ij] * energy >= model.pE_charged_1[ij])

    model.c_profitable_stations = ConstraintList()
    model.c12 = ConstraintList()
    model.installed_infrastructure = ConstraintList()
    # install existing infrastructure
    if input_existing_infrastructure:
        remaining_demand_0, remaining_demand_1, installed_cs, total_poles = calculate_remaining_demand()
        print('remaining_demand', sum(remaining_demand_0) + sum(remaining_demand_1))
        for ij in model.IDX_2:
            name = dir_names[ij]
            direc = dir_directions[ij]
            highway = dir_highways[ij]
            extract_dir_0 = dir_0[
                (
                        (dir_0[col_rest_area_name] == name)
                        & (dir_0[col_directions] == direc)
                        & (dir_0[col_highway] == highway)
                )
            ].index.to_list()
            extract_dir_1 = dir_1[
                (
                        (dir_1[col_rest_area_name] == name)
                        & (dir_1[col_directions] == direc)
                        & (dir_1[col_highway] == highway)
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
                if installed_cs[ij] > 0:
                    model.installed_infrastructure.add(model.pXi[ij] == 1)
                    model.installed_infrastructure.add(model.pYi_dir_0[model.IDX_0[ind_0]] + model.pYi_dir_1[model.IDX_1[ind_1]] >= total_poles[ij])
                else:
                    current_demand = (remaining_demand_0[model.IDX_0[model.IDX_0[ind_0]]] + remaining_demand_1[
                        model.IDX_1[ind_1]]) * eta * mu
                    is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                    if is_profitable:
                        model.c_profitable_stations.add(model.pXi[ij] == 1)
                        model.c_profitable_stations.add(
                            model.pYi_dir_0[model.IDX_0[ind_0]] + model.pYi_dir_1[model.IDX_1[ind_1]] >= 1)

            elif len(extract_dir_0) > 0:
                ind_0 = extract_dir_0[0]
                model.c12.add(
                    model.pYi_dir_0[model.IDX_0[ind_0]]
                    <= g * model.pXi[ij]
                )
                if installed_cs[ij] > 0:
                    model.installed_infrastructure.add(model.pXi[ij] == 1)
                    model.installed_infrastructure.add(model.pYi_dir_0[ind_0] >= total_poles[ij])
                else:
                    current_demand = (remaining_demand_0[model.IDX_0[model.IDX_0[ind_0]]]) * eta * mu
                    is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                    if is_profitable:
                        model.c_profitable_stations.add(model.pXi[ij] == 1)
                        model.c_profitable_stations.add(
                            model.pYi_dir_0[model.IDX_0[ind_0]] >= 1)
            elif len(extract_dir_1) > 0:
                ind_1 = extract_dir_1[0]
                model.c12.add(
                    model.pYi_dir_1[model.IDX_1[ind_1]]
                    <= g * model.pXi[ij]
                )
                if installed_cs[ij] > 0:
                    model.installed_infrastructure.add(model.pXi[ij] == 1)
                    model.installed_infrastructure.add(model.pYi_dir_1[ind_1] >= total_poles[ij])
                else:
                    current_demand = (remaining_demand_1[
                        model.IDX_1[ind_1]]) * eta * mu
                    is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                    if is_profitable:
                        model.c_profitable_stations.add(model.pXi[ij] == 1)
                        model.c_profitable_stations.add(model.pYi_dir_1[model.IDX_1[ind_1]] >= 1)

    else:
        for ij in model.IDX_2:
            name = dir_names[ij]
            direc = dir_directions[ij]
            highway = dir_highways[ij]
            extract_dir_0 = dir_0[
                (
                    (dir_0[col_rest_area_name] == name)
                    & (dir_0[col_directions] == direc)
                    & (dir_0[col_highway] == highway)
                )
            ].index.to_list()
            extract_dir_1 = dir_1[
                (
                    (dir_1[col_rest_area_name] == name)
                    & (dir_1[col_directions] == direc)
                    & (dir_1[col_highway] == highway)
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
                current_demand = (energy_demand_0[model.IDX_0[ind_0]] + energy_demand_1[model.IDX_1[ind_1]]) * eta * mu
                is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                if is_profitable:
                    model.c_profitable_stations.add(model.pXi[ij] == 1)
                    model.c_profitable_stations.add((model.pXi[ij] == 1) >>
                                    (model.pYi_dir_0[model.IDX_0[ind_0]] + model.pYi_dir_1[model.IDX_1[ind_1]] >= 1))

            elif len(extract_dir_0) > 0:
                ind_0 = extract_dir_0[0]
                model.c12.add(model.pYi_dir_0[model.IDX_0[ind_0]] <= g * model.pXi[ij])
                current_demand = energy_demand_0[model.IDX_0[ind_0]] * eta * mu
                is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                if is_profitable:
                    model.c_profitable_stations.add(model.pXi[ij] == 1)
                    model.c_profitable_stations.add(model.pYi_dir_0[model.IDX_0[ind_0]] >= 1)

            elif len(extract_dir_1) > 0:
                ind_1 = extract_dir_1[0]
                model.c12.add(model.pYi_dir_1[model.IDX_1[ind_1]] <= g * model.pXi[ij])
                current_demand = energy_demand_1[model.IDX_1[ind_1]] * eta * mu
                is_profitable, num_var = profitable(i, ec, T, e_tax, cfix, cvar, current_demand, energy)
                if is_profitable:
                    model.c_profitable_stations.add(model.pXi[ij] == 1)
                    model.c_profitable_stations.add(model.pYi_dir_1[model.IDX_1[ind_1]] >= 1)

    # zero-constraints
    model.c14 = ConstraintList()
    for ij in model.IDX_0:
        model.c14.add(model.pE_charged_0[ij] >= 0)
        model.c14.add(model.pE_input_0[ij] >= 0)
        model.c14.add(model.pE_output_0[ij] >= 0)

    for ij in model.IDX_1:
        model.c14.add(model.pE_charged_1[ij] >= 0)
        model.c14.add(model.pE_input_1[ij] >= 0)
        model.c14.add(model.pE_output_1[ij] >= 0)

    # ------------------------------------------------- objective -------------------------------------------------
    # maximization of revenue during the observed observation period

    model.obj = Objective(
        expr=(
            sum(model.pE_charged_0[n] * (e_tax) for n in model.IDX_0)
            + sum(model.pE_charged_1[n] * (e_tax) for n in model.IDX_1)
            + (
                1
                / RBF
                * 1
                / 365
                * (
                        sum(model.pXi[n] for n in model.IDX_2) * cfix
                        + (
                        sum(model.pYi_dir_0[n] for n in model.IDX_0)
                        + sum(model.pYi_dir_1[n] for n in model.IDX_1)
                    )
                        * cvar
                )
            )
        ),
        sense=minimize,
    )

    # ------------------------------------------------- solution --------------------------------------------------

    opt = SolverFactory("gurobi")
    opt_success = opt.solve(model)
    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")

    # -------------------------------------------------- results --------------------------------------------------

    pYi_dir_0 = np.array([model.pYi_dir_0[n].value for n in model.IDX_0])
    pYi_dir_1 = np.array([model.pYi_dir_1[n].value for n in model.IDX_1])
    pXi = np.array([model.pXi[n].value for n in model.IDX_2])
    pXi_dir_0 = np.where(pYi_dir_0 > 0, 1, 0)
    pXi_dir_1 = np.where(pYi_dir_1 > 0, 1, 0)
    pE_input_0 = np.array([model.pE_input_0[n].value for n in model.IDX_0])
    pE_input_1 = np.array([model.pE_input_1[n].value for n in model.IDX_1])
    pE_output_0 = np.array([model.pE_output_0[n].value for n in model.IDX_0])
    pE_output_1 = np.array([model.pE_output_1[n].value for n in model.IDX_1])
    pE_charged_0 = np.array([model.pE_charged_0[n].value for n in model.IDX_0])
    pE_charged_1 = np.array([model.pE_charged_1[n].value for n in model.IDX_1])
    umsatz = sum(pE_charged_0 * ec) + sum(pE_charged_1 * ec)

    print(colored("Total revenue per year: € " + str(umsatz * 365), "green"))
    print(
        colored(
            "Total Profit per year (obj. fun.): € " + str(model.obj.value() * 365),
            "green",
        )
    )
    print(
        colored(
            "Total Profit per year for one charing pole: € "
            + str(model.obj.value() * 365 / (sum(pYi_dir_0) + sum(pYi_dir_1))),
            "green",
        )
    )
    print(
        colored(
            "Total amount of charging poles: " + str(sum(pYi_dir_0) + sum(pYi_dir_1)),
            "green",
        )
    )

    # creating output file
    # merging both dataframes to singular table with all resting areas and calculated data
    output_cols = [
        col_highway,
        col_rest_area_name,
        col_position,
        col_directions,
        col_type,
        col_traffic_flow,
        col_energy_demand,
    ]
    singular_info = [
        col_highway,
        col_rest_area_name,
        col_position,
        col_directions,
        col_type,
        "pXi_dir",
    ]
    output_dir_0 = dir_0[output_cols]
    output_dir_1 = dir_1[output_cols]
    output_dir_0["pXi_dir"] = pXi_dir_0
    output_dir_1["pXi_dir"] = pXi_dir_1
    output_dir_0["pYi_dir_0"] = pYi_dir_0
    output_dir_1["pYi_dir_1"] = pYi_dir_1
    output_dir_0["pE_charged_0"] = pE_charged_0
    output_dir_1["pE_charged_1"] = pE_charged_1
    output_dir_0["pE_input_0"] = pE_input_0
    output_dir_1["pE_input_1"] = pE_input_1
    output_dir_0["pE_output_0"] = pE_output_0
    output_dir_1["pE_output_1"] = pE_output_1
    model_variable_names = ["pYi_dir", "pE_charged", "pE_input", "pE_output"]

    all_info_df = output_dir_0.append(output_dir_1)
    all_info_df.index = range(0, len(all_info_df))
    all_info_df = all_info_df.fillna(0.0)
    output_dataframe = pd.DataFrame()
    dir_0_names = dir_0[col_rest_area_name].to_list()
    dir_1_names = dir_1[col_rest_area_name].to_list()
    all_ra_names = list(set(dir_0_names + dir_1_names))
    output_dataframe.index = all_ra_names
    key_list = all_info_df.keys().to_list()
    for name in all_ra_names:
        extract_name_df = all_info_df[all_info_df[col_rest_area_name] == name]
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

    output_dataframe = output_dataframe.fillna(0.0)

    output_filename = (
        "results/" + time_of_optimization + "_optimization_result_charging_stations.csv"
    )
    output_dataframe = output_dataframe.sort_values(by=[col_highway, col_position])
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
        + ";  hours_of_constant_charging="
        + str(hours_of_constant_charging)
        + ";  i="
        + str(i)
        + ";  T="
        + str(T)
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
        cars_per_day=cars_per_day,
        energy=energy,
        specific_demand=specific_demand,
    )

    return sum(pXi), sum(pYi_dir_0) + sum(pYi_dir_1), model.obj.value() * 365


if __name__ == "__main__":
    optimization(input_existing_infrastructure=False)
