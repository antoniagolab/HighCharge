"""

Created by Antonia Golab

Optimization of charging station allocation

last edit: 2021/12/08

"""


# TODO: reformat and make modular !!


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
    specific_demand=specific_demand,
    input_existing_infrastructure=False,
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
    model.pXi = Var(model.IDX_2, within=Binary)
    # model.X = Var(model.IDX_0, within=Binary)

    # integer variables indicating number of charging poles at a charging station
    model.pYi_dir_0 = Var(model.IDX_0, within=Integers)
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    # a fictitious energy load is passed on from ra to ra along a highway which (pE_Zu) which is the difference between
    # energy demand and covered energy demand at a resting area (pE_Laden); the remaining energy is passed on to the
    # next ra (pE_Ab)
    model.pE_input_0 = Var(model.IDX_0, model.IDX_0)
    model.pE_input_1 = Var(model.IDX_1, model.IDX_1)
    model.pE_output_0 = Var(model.IDX_0, model.IDX_0)
    model.pE_output_1 = Var(model.IDX_1, model.IDX_1)
    model.pE_charged_0 = Var(model.IDX_0, model.IDX_0)
    model.pE_charged_1 = Var(model.IDX_1, model.IDX_1)

    cars_per_day = hours_of_constant_charging / (acc / charging_capacity)
    energy = acc * cars_per_day  # (kWh) charging energy per day by one charging pole

    # ------------------------------------------------- constraints -------------------------------------------------

    # the pE_Zu needs to be 0 at each first ra as it is assumed that no energy demand is previously covered;
    # all energy demand is constrained to be covered at the last ra on a highway
    model.c1 = ConstraintList()

    # direction == 0
    for ij in model.IDX_0:
        for kl in model.IDX_0:
            if mask_0[ij, kl] == 0:
                model.c1.add(model.pE_input_0[ij, kl] == 0)
                model.c1.add(model.pE_output_0[ij, kl] == 0)
                model.c1.add(model.pE_charged_0[ij, kl] == 0)
            if enum_0[ij, kl] == 1:
                model.c1.add(model.pE_input_0[ij, kl] == 0)
            if enum_0[ij, kl] == max(
                enum_0[
                    ij,
                ]
            ):
                model.c1.add(model.pE_output_0[ij, kl] == 0)

    # direction == 1
    for ij in model.IDX_1:
        for kl in model.IDX_1:
            if mask_1[ij, kl] == 0:
                model.c1.add(model.pE_input_1[ij, kl] == 0)
                model.c1.add(model.pE_output_1[ij, kl] == 0)
                model.c1.add(model.pE_charged_1[ij, kl] == 0)
            if enum_1[ij, kl] == 1:
                model.c1.add(model.pE_input_1[ij, kl] == 0)
            if enum_1[ij, kl] == max(
                enum_1[
                    ij,
                ]
            ):
                model.c1.add(model.pE_output_1[ij, kl] == 0)

    # constraints defining the passing on of the fictitious energy
    model.c2 = ConstraintList()

    for ij in model.IDX_0:
        current_enum = enum_0[
            ij,
        ]
        ind_sortation_list = np.argsort(current_enum)
        ind_sortation_list = [
            kl for kl in ind_sortation_list if not current_enum[kl] == 0
        ]
        for mn in range(1, len(ind_sortation_list)):
            model.c2.add(
                model.pE_output_0[ij, ind_sortation_list[mn - 1]]
                == model.pE_input_0[ij, ind_sortation_list[mn]]
            )

    for ij in model.IDX_1:
        current_enum = enum_1[
            ij,
        ]
        ind_sortation_list = np.argsort(current_enum)
        ind_sortation_list = [
            kl for kl in ind_sortation_list if not current_enum[kl] == 0
        ]
        for mn in range(1, len(ind_sortation_list)):
            model.c2.add(
                model.pE_output_1[ij, ind_sortation_list[mn - 1]]
                == model.pE_input_1[ij, ind_sortation_list[mn]]
            )

    for ij in model.IDX_0:
        model.c2.add(
            sum(model.pE_charged_0[ij, kl] for kl in model.IDX_0)
            == energy_demand_matrix_0[ij, ij]
        )

    for ij in model.IDX_1:
        model.c2.add(
            sum(model.pE_charged_1[ij, kl] for kl in model.IDX_1)
            == energy_demand_matrix_1[ij, ij]
        )

    # defining the relationship between energy demand, demand coverage and net energy demand
    model.c7 = ConstraintList()

    # relationship between individual matrix elements
    for ij in model.IDX_0:
        for kl in model.IDX_0:
            model.c7.add(
                model.pE_charged_0[ij, kl]
                - energy_demand_matrix_0[ij, kl]
                - model.pE_input_0[ij, kl]
                + model.pE_output_0[ij, kl]
                == 0
            )

    for ij in model.IDX_1:
        for kl in model.IDX_1:
            model.c7.add(
                model.pE_charged_1[ij, kl]
                - energy_demand_matrix_1[ij, kl]
                - model.pE_input_1[ij, kl]
                + model.pE_output_1[ij, kl]
                == 0
            )

    # relating overall energy movement

    model.c8 = ConstraintList()
    for ij in model.IDX_0:
        model.c8.add(
            sum(model.pE_charged_0[kl, ij] for kl in model.IDX_0)
            - energy_demand_matrix_0[ij, ij]
            - sum(model.pE_input_0[kl, ij] for kl in model.IDX_0)
            + sum(model.pE_output_0[kl, ij] for kl in model.IDX_0)
            == 0
        )

    for ij in model.IDX_1:
        model.c8.add(
            sum(model.pE_charged_1[kl, ij] for kl in model.IDX_1)
            - energy_demand_matrix_1[ij, ij]
            - sum(model.pE_input_1[kl, ij] for kl in model.IDX_1)
            + sum(model.pE_output_1[kl, ij] for kl in model.IDX_1)
            == 0
        )

    # total demand needs to be covered
    model.c10 = ConstraintList()
    for ij in model.IDX_0:
        model.c10.add(
            sum([model.pE_charged_0[ij, kl] for kl in model.IDX_0])
            == energy_demand_matrix_0[ij, ij]
        )

    for ij in model.IDX_1:
        model.c10.add(
            sum([model.pE_charged_1[ij, kl] for kl in model.IDX_1])
            == energy_demand_matrix_1[ij, ij]
        )

    # TODO: to correctly regard already installed charging infrastructure, it is important to implement two additional
    #   variables (X installed, Y installed) with are also introduced to the objective function as some kind of price
    #   reduction or so because these are charging station possibilities "for free"; here it has to be specifically
    #   evaluated if this is current handling of charging infrastructure is enough and we just have to insert a minus
    #   term to the objective function

    model.c_profitable_stations = ConstraintList()
    model.c12 = ConstraintList()
    model.installed_infrastructure = ConstraintList()
    dir_names = dir[col_rest_area_name].to_list()
    dir_directions = dir[col_directions].to_list()
    dir_highways = dir[col_highway].to_list()

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
            )

        elif len(extract_dir_0) > 0:

            ind_0 = extract_dir_0[0]
            model.c12.add(model.pYi_dir_0[model.IDX_0[ind_0]] <= g * model.pXi[ij])
            model.c12.add(
                model.pYi_dir_0[model.IDX_0[ind_0]] * energy
                >= sum(
                    [model.pE_charged_0[kl, model.IDX_0[ind_0]] for kl in model.IDX_0]
                )
            )

        elif len(extract_dir_1) > 0:

            ind_1 = extract_dir_1[0]
            model.c12.add(model.pYi_dir_1[model.IDX_1[ind_1]] <= g * model.pXi[ij])
            model.c12.add(
                model.pYi_dir_1[model.IDX_1[ind_1]] * energy
                >= sum(
                    [model.pE_charged_1[kl, model.IDX_1[ind_1]] for kl in model.IDX_1]
                )
            )

    # zero-constraints
    model.c14 = ConstraintList()
    for ij in model.IDX_0:
        for kl in model.IDX_0:
            model.c14.add(model.pE_charged_0[ij, kl] >= 0)
            model.c14.add(model.pE_input_0[ij, kl] >= 0)
            model.c14.add(model.pE_output_0[ij, kl] >= 0)

    # zero-constraints
    for ij in model.IDX_1:
        for kl in model.IDX_1:
            model.c14.add(model.pE_charged_1[ij, kl] >= 0)
            model.c14.add(model.pE_input_1[ij, kl] >= 0)
            model.c14.add(model.pE_output_1[ij, kl] >= 0)

    # ------------------------------------------------- objective -------------------------------------------------
    # minimization of installation costs

    model.obj = Objective(
        expr=(
            (
                1
                / RBF
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
    pE_input_0_m = []
    pE_input_1_m = []
    pE_output_0_m = []
    pE_output_1_m = []
    pE_charged_0_m = []
    pE_charged_1_m = []

    for ij in model.IDX_0:
        pE_charged_0_m.append([model.pE_charged_0[ij, kl].value for kl in model.IDX_0])
        pE_input_0_m.append([model.pE_input_0[ij, kl].value for kl in model.IDX_0])
        pE_output_0_m.append([model.pE_output_0[ij, kl].value for kl in model.IDX_0])

    for ij in model.IDX_1:
        pE_input_1_m.append([model.pE_input_1[ij, kl].value for kl in model.IDX_1])
        pE_charged_1_m.append([model.pE_charged_1[ij, kl].value for kl in model.IDX_1])
        pE_output_1_m.append([model.pE_output_1[ij, kl].value for kl in model.IDX_1])

    pE_charged_0_m = np.array(pE_charged_0_m)
    pE_input_0_m = np.array(pE_input_0_m)
    pE_output_0_m = np.array(pE_output_0_m)
    pE_output_1_m = np.array(pE_output_1_m)
    pE_input_1_m = np.array(pE_input_1_m)
    pE_charged_1_m = np.array(pE_charged_1_m)

    pE_charged_0 = np.sum(pE_charged_0_m, axis=0)
    pE_charged_1 = np.sum(pE_charged_1_m, axis=0)
    pE_input_0 = np.sum(pE_input_0_m, axis=0)
    pE_input_1 = np.sum(pE_input_1_m, axis=0)
    pE_output_0 = np.sum(pE_output_0_m, axis=0)
    pE_output_1 = np.sum(pE_output_1_m, axis=0)

    print(
        colored(
            "Total Profit installation costs: € " + str(model.obj.value()),
            "green",
        )
    )
    print(
        colored(
            "Total amount of charging poles: " + str(sum(pYi_dir_0) + sum(pYi_dir_1)),
            "green",
        )
    )
    print(colored("Total amount of charging stations: " + str(sum(pXi)), "green"))
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

    output_dataframe["pXi"] = np.where((output_dataframe.pYi_dir > 0), 1, 0)
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
