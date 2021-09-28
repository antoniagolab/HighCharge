"""

Created by Antonia Golab

Optimization of charging station allocation

last edit: 2021/12/08


TODO:
    - if waiting time > 10min to charge, then remove parking places (3)
    - recalculate demand based on this (4)
    - make it possible to read status quo and set constraints based on these
    - recalculate demand based on this
    - evaluate whether charging stations are to be placed based on profitability
    - cost minimum objective function
    - some small e-charge which is to be removed? -> some small amount for which it is not profitable to be covered


TODO: comparison of different optimization strategies: placing a charging station everywhere where it is profitable
    based on the local demand - then (A) minimizing total costs OR (B) setting for each charging station a profitability
    constraint but maximizing the amount of charging stations
"""


import pandas as pd
import numpy as np
from optimization_parameters import *
from pyomo.environ import *
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
    specific_demand=specific_demand,
):
    """
    Constraint and objective function definition + solution of optimization using parameters defined in
    optimization_parameters.py

    :return:
    """
    # removing parking places from possible locations if charging time < 10 min
    # and recalculate demands
    asfinag_type_list_0 = dir_0[col_type].to_list()
    asfinag_type_list_1 = dir_1[col_type].to_list()
    dir_0_inds = dir_0.index.to_list()
    dir_1_inds = dir_1.index.to_list()
    if time_of_charging > max_charging_at_pp:
            for name in highway_names:
                extract_dir_0 = dir_0[dir_0[col_highway] == name]
                extract_dir_1 = dir_1[dir_1[col_highway] == name]
                for ij in range(0, len(extract_dir_0)-1):
                    if asfinag_type_list_0[ij] == 2:
                        energy_demand_0[ij + 1] = energy_demand_0[ij] + energy_demand_0[ij+1]

                for ij in range(0, len(extract_dir_1)-1):
                    if asfinag_type_list_1[ij] == 2:
                        energy_demand_1[ij + 1] = energy_demand_1[ij] + energy_demand_1[ij + 1]


            refac_dir_0 = dir_0
            refac_dir_1 = dir_1
            refac_dir_0[col_energy_demand] = energy_demand_0
            refac_dir_1[col_energy_demand] = energy_demand_1
            refac_dir_0 = refac_dir_0[~(refac_dir_0[col_type] == 2)]
            refac_dir_1 = refac_dir_1[~(refac_dir_1[col_type] == 2)]
            refac_energy_demand_0 = [d * tf_at_peak for d in refac_dir_0[
                col_energy_demand].to_list()]  # (kWh/d) energy demand at each rest area per day
            refac_energy_demand_1 = [d * tf_at_peak for d in refac_dir_1[col_energy_demand].to_list()]
    else:
        refac_dir_0 = dir_0
        refac_dir_1 = dir_1
        refac_energy_demand_0 = [d * tf_at_peak for d in refac_dir_0[
            col_energy_demand].to_list()]  # (kWh/d) energy demand at each rest area per day
        refac_energy_demand_1 = [d * tf_at_peak for d in refac_dir_1[col_energy_demand].to_list()]

    refac_dir_0["ID"] = range(0, len(refac_dir_0))
    refac_dir_0 = refac_dir_0.set_index("ID")

    refac_dir_1["ID"] = range(0, len(refac_dir_1))
    refac_dir_1 = refac_dir_1.set_index("ID")
    model = ConcreteModel()  # Model initialization

    n0 = len(refac_dir_0)
    n1 = len(refac_dir_1)

    print(n0, n1)

    # --------------------------------------------- variable definition ---------------------------------------------
    # driving directions "normal" (=0) and "inverse" (=1) are treated separately throughout the optimization model
    # charging demand at each resting area is then summed up if it is for both directions
    model.IDX_0 = range(0, n0)
    model.IDX_1 = range(0, n1)
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

    cars_per_hour = hourly_res / (acc / charging_capacity)
    energy = acc * cars_per_hour  # (kWh) charging energy per day by one charging pole
    # ------------------------------------------------- constraints -------------------------------------------------

    # the pE_Zu needs to be 0 at each first ra as it is assumed that no energy demand is previously covered;
    # all energy demand is constrained to be covered at the last ra on a highway
    model.c1 = ConstraintList()

    firsts_0 = refac_dir_0[refac_dir_0["first"]].index
    for ind in firsts_0:
        model.c1.add(model.pE_input_0[ind] == 0)
    lasts_0 = refac_dir_0[refac_dir_0["last"]].index
    for ind in lasts_0:
        model.c1.add(model.pE_output_0[ind] == 0)

    firsts_1 = refac_dir_1[refac_dir_1["first"]].index
    for ind in firsts_1:
        model.c1.add(model.pE_input_1[ind] == 0)
    lasts_1 = refac_dir_1[refac_dir_1["last"]].index
    for ind in lasts_1:
        model.c1.add(model.pE_output_1[ind] == 0)

    # constraints defining the passing on of the fictitious energy
    model.c2 = ConstraintList()
    for ij in range(1, n0):
        model.c2.add(model.pE_input_0[ij] == model.pE_output_0[ij - 1])

    for ij in range(1, n1):
        model.c2.add(model.pE_input_1[ij] == model.pE_output_1[ij - 1])

    # constraining the fictitiously passed on energy to ensure evenly distributed charging network

    dir_0_highways = refac_dir_0[col_highway].to_list()
    model.c3 = ConstraintList()
    for ij in model.IDX_0:
        highway = dir_0_highways[ij]
        model.c3.add(
            model.pE_input_0[ij] / (traffic_flows_dir_0[ij] * specific_demand * tf_at_peak * eta * mu)
            <= maximum_dist_between_charging_stations
        )
        model.c3.add(
            model.pE_output_0[ij] / (traffic_flows_dir_0[ij] * specific_demand * tf_at_peak * eta * mu)
            <= maximum_dist_between_charging_stations
        )

    dir_1_highways = refac_dir_1[col_highway].to_list()
    model.c4 = ConstraintList()
    for ij in model.IDX_1:
        highway = dir_1_highways[ij]
        model.c4.add(
            model.pE_input_1[ij] / (traffic_flows_dir_1[ij] * specific_demand * tf_at_peak * eta * mu)
            <= maximum_dist_between_charging_stations
        )
        model.c4.add(
            model.pE_output_1[ij] / (traffic_flows_dir_1[ij] * specific_demand * tf_at_peak * eta * mu)
            <= maximum_dist_between_charging_stations
        )

    # defining the relationship between energy demand, demand coverage and net energy demand
    model.c7 = ConstraintList()
    for ij in model.IDX_0:
        model.c7.add(
            model.pE_charged_0[ij]
            - refac_energy_demand_0[ij] * eta * mu
            - model.pE_input_0[ij]
            + model.pE_output_0[ij]
            == 0
        )

    model.c8 = ConstraintList()
    for ij in model.IDX_1:
        model.c8.add(
            model.pE_charged_1[ij]
            - refac_energy_demand_1[ij] * eta * mu
            - model.pE_input_1[ij]
            + model.pE_output_1[ij]
            == 0
        )

    # defining how much energy demand a charging station is able to cover
    model.c9 = ConstraintList()
    for ij in model.IDX_0:
        model.c9.add(model.pYi_dir_0[ij] * energy >= model.pE_charged_0[ij])

    model.c10 = ConstraintList()
    for ij in model.IDX_1:
        model.c10.add(model.pYi_dir_1[ij] * energy >= model.pE_charged_1[ij])

    # setting a maximum number at each station + defining relation between the binary variable defining whether a
    # resting area has a charging station and the number of charging poles at one

    dir_names = dir[col_rest_area_name].to_list()
    dir_directions = dir[col_directions].to_list()
    dir_highways = dir[col_highway].to_list()
    model.c12 = ConstraintList()
    for ij in model.IDX_2:
        name = dir_names[ij]
        direc = dir_directions[ij]
        highway = dir_highways[ij]
        extract_dir_0 = refac_dir_0[
            (
                (refac_dir_0[col_rest_area_name] == name)
                & (refac_dir_0[col_directions] == direc)
                & (refac_dir_0[col_highway] == highway)
            )
        ].index.to_list()
        extract_dir_1 = refac_dir_1[
            (
                (refac_dir_1[col_rest_area_name] == name)
                & (refac_dir_1[col_directions] == direc)
                & (refac_dir_1[col_highway] == highway)
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
        if len(extract_dir_0) > 0:
            ind_0 = extract_dir_0[0]
            model.c12.add(model.pYi_dir_0[model.IDX_0[ind_0]] <= g * model.pXi[ij])
        if len(extract_dir_1) > 0:
            ind_1 = extract_dir_1[0]
            model.c12.add(model.pYi_dir_1[model.IDX_1[ind_1]] <= g * model.pXi[ij])

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
    # maximization of profit during the observed observation period
    # TODO: change to minimization of costs

    model.obj = Objective(
        expr=(
            sum(model.pE_charged_0[n] * (ec - e_tax) for n in model.IDX_0) * 24 * tf_at_peak
            + sum(model.pE_charged_1[n] * (ec - e_tax) for n in model.IDX_1) * 24 * tf_at_peak
            - (
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
                    * cvar1
                )
            )
        ),
        sense=maximize,
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
    output_dir_0 = refac_dir_0[output_cols]
    output_dir_1 = refac_dir_1[output_cols]
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
    dir_0_names = refac_dir_0[col_rest_area_name].to_list()
    dir_1_names = refac_dir_1[col_rest_area_name].to_list()
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
        + str(cvar1)
        + ";  eta="
        + str(eta)
        + "; mu="
        + str(mu)
        + "; tf_at_peak="
        + str(tf_at_peak)
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
        cars_per_day=cars_per_hour,
        energy=energy,
        specific_demand=specific_demand,
    )

    return sum(pXi), sum(pYi_dir_0) + sum(pYi_dir_1), model.obj.value() * 365


if __name__ == "__main__":
    optimization()
