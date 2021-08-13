"""

Created by Antonia Golab

Optimization of charging station allocation

last edit: 2021/12/08

"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from optimization_parameters import *
import time
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def optimization():
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

    # binary variables stating whether a charging station is installed at given ra (=resting area)
    model.pXi_dir_0 = Var(model.IDX_0, within=Binary)
    model.pXi_dir_1 = Var(model.IDX_1, within=Binary)

    # integer variables indicating number of charging poles at a charging station
    model.pYi_dir_0 = Var(
        model.IDX_0,
        within=Integers,
    )
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    # a fictitious energy load is passed on from ra to ra along a highway which (pE_Zu) which is the difference between
    # energy demand and covered energy demand at a resting area (pE_Laden); the remaining energy is passed on to the
    # next ra (pE_Ab)
    model.pE_input_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_input_1 = Var(model.IDX_1, within=NonNegativeReals)
    model.pE_output_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_output_1 = Var(model.IDX_1, within=NonNegativeReals)
    model.pE_charged_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_charged_1 = Var(model.IDX_1, within=NonNegativeReals)

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

    # constraints defining the passing on of the fictitious energy
    model.c2 = ConstraintList()
    for ij in range(1, n0):
        model.c2.add(model.pE_input_0[ij] == model.pE_output_0[ij - 1])

    for ij in range(1, n1):
        model.c2.add(model.pE_input_1[ij] == model.pE_output_1[ij - 1])

    # constraining the fictitiously passed on energy to ensure evenly distributed charging network
    model.c3 = ConstraintList()
    for ij in model.IDX_0:
        model.c3.add(model.pE_input_0[ij] <= e_average)

    model.c4 = ConstraintList()
    for ij in model.IDX_1:
        model.c4.add(model.pE_input_1[ij] <= e_average)

    model.c5 = ConstraintList()
    for ij in model.IDX_0:
        model.c5.add(model.pE_output_0[ij] <= e_average)

    model.c6 = ConstraintList()
    for ij in model.IDX_1:
        model.c6.add(model.pE_output_1[ij] <= e_average)

    # defining the relationship between energy demand, demand coverage and net energy demand
    model.c7 = ConstraintList()
    for ij in model.IDX_0:
        model.c7.add(
            model.pE_charged_0[ij]
            - energy_demand_0[ij] * eta
            - model.pE_input_0[ij]
            + model.pE_output_0[ij]
            == 0
        )

    model.c8 = ConstraintList()
    for ij in model.IDX_1:
        model.c8.add(
            model.pE_charged_1[ij]
            - energy_demand_1[ij] * eta
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

    # installing a constraint preferring charging station at ras for both directions
    model.c11 = ConstraintList()
    inds_pref = [
        ij
        for ij in range(0, len(directions_0))
        if directions_0[ij] == 2 and energy_demand_0[ij] > energy / eta
    ]
    for ind in inds_pref:
        name = dir_0[col_rest_area_name].to_list()[ind]
        dir_1_inds = dir_1[dir_1[col_rest_area_name] == name].index
        model.c11.add(model.pXi_dir_0[ind] == 1)
        model.c11.add(model.pYi_dir_0[ind] >= 1)
        model.c11.add(model.pXi_dir_1[dir_1_inds[0]] == 1)
        model.c11.add(model.pYi_dir_1[dir_1_inds[0]] >= 1)
    count = len(inds_pref)
    # setting a maximum number at each station + defining relation between the binary variable defining whether a
    # resting area has a charging station and the number of charging poles at one
    model.c12 = ConstraintList()
    for ij in model.IDX_0:
        model.c12.add(model.pYi_dir_0[ij] <= g * model.pXi_dir_0[ij])

    model.c13 = ConstraintList()
    for ij in model.IDX_1:
        model.c12.add(model.pYi_dir_1[ij] <= g * model.pXi_dir_1[ij])

    # ------------------------------------------------- objective -------------------------------------------------
    # maximization of revenue during the observed observation period

    model.obj = Objective(
        expr=(
            sum(model.pE_charged_0[n] * (ec - e_tax) for n in model.IDX_0)
            + sum(model.pE_charged_1[n] * (ec - e_tax) for n in model.IDX_1)
            - (
                1
                / RBF
                * 1
                / 365
                * (
                    (
                        sum(model.pXi_dir_0[n] for n in model.IDX_0)
                        + sum(model.pYi_dir_1[n] for n in model.IDX_1)
                        - count
                    )
                    * cfix
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
    pXi_dir_0 = np.array([model.pXi_dir_0[n].value for n in model.IDX_0])
    pXi_dir_1 = np.array([model.pXi_dir_1[n].value for n in model.IDX_1])
    pE_input_0 = np.array([model.pE_input_0[n].value for n in model.IDX_0])
    pE_input_1 = np.array([model.pE_input_1[n].value for n in model.IDX_1])
    pE_output_0 = np.array([model.pE_output_0[n].value for n in model.IDX_0])
    pE_output_1 = np.array([model.pE_output_1[n].value for n in model.IDX_1])
    pE_charged_0 = np.array([model.pE_charged_0[n].value for n in model.IDX_0])
    pE_charged_1 = np.array([model.pE_charged_1[n].value for n in model.IDX_1])
    umsatz = sum(pE_charged_0 * ec) + sum(pE_charged_1 * ec)

    print(f"Total revenue per year: € {umsatz*365}")
    print(f"Total amount of charging poles: {sum(pYi_dir_0)+sum(pYi_dir_1)}")

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
        + str(cvar1)
        + ";  eta="
        + str(eta)
        + ";  cars="
        + str(cars)
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

    return model.obj.values()


if __name__ == "__main__":
    optimization()
