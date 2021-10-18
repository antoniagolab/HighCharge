import numpy as np
import pandas as pd
from optimization_parameters import *
from pyomo.environ import *
import time

def calculate_remaining_demand():

    rest_areas_0 = pd.read_csv('data/rest_areas_1_charging_stations.csv')
    rest_areas_0.cs_below_50kwh = rest_areas_0.cs_below_50kwh.fillna(False)
    rest_areas_0 = pd.merge(rest_areas_0, dir_0[[col_highway, col_rest_area_name, col_directions, col_energy_demand,
                                                 col_position, col_distance]], on=[col_highway, col_rest_area_name, col_directions])
    rest_areas_0 = rest_areas_0.sort_values(by=[col_highway, col_distance])
    # rest_areas_0 = rest_areas_0[rest_areas_0[col_highway] == "A1"]
    rest_areas_1 = pd.read_csv('data/rest_areas_0_charging_stations.csv')
    rest_areas_1.cs_below_50kwh = rest_areas_1.cs_below_50kwh.fillna(False)
    rest_areas_1 = pd.merge(rest_areas_1, dir_1[[col_highway, col_rest_area_name, col_directions, col_energy_demand,
                                                 col_position, col_distance]]
                            , on=[col_highway, col_rest_area_name, col_directions])
    rest_areas_1 = rest_areas_1.sort_values(by=[col_highway, col_distance])
    # rest_areas_1 = rest_areas_1[rest_areas_1[col_highway] == "A1"]

    highway_names = list(set(rest_areas_0[col_highway].to_list() + rest_areas_1[col_highway].to_list()))

    # find highways with no charging stations which need to be removed as for them no charging infrastructure exists
    highway_no_infrastructure = []
    for name in highway_names:
        extract_ra_0 = rest_areas_0[((rest_areas_0.highway == name) & (rest_areas_0[col_has_cs] == True))]
        extract_ra_1 = rest_areas_1[((rest_areas_1.highway == name) & (rest_areas_1[col_has_cs] == True))]
        if len(extract_ra_0) == 0 and len(extract_ra_1) == 0:
            highway_no_infrastructure.append(name)

    # rest_areas_0 = rest_areas_0[~rest_areas_0[col_highway].isin(highway_no_infrastructure)]
    rest_areas_0['ID'] = range(0, len(rest_areas_0))
    rest_areas_0 = rest_areas_0.set_index("ID")
    # rest_areas_1 = rest_areas_1[~rest_areas_1[col_highway].isin(highway_no_infrastructure)]
    rest_areas_1['ID'] = range(0, len(rest_areas_1))
    rest_areas_1 = rest_areas_1.set_index("ID")

    # one dataframe with all rest areas
    rest_areas = rest_areas_0.append(rest_areas_1).drop_duplicates(subset=[col_rest_area_name, col_directions])
    rest_areas = rest_areas.sort_values(by=[col_highway, col_position, col_directions])
    rest_areas['ID'] = range(0, len(rest_areas))
    rest_areas = rest_areas.set_index("ID")

    energy_demand_0 = rest_areas_0[col_energy_demand].to_list()
    energy_demand_1 = rest_areas_1[col_energy_demand].to_list()

    model = ConcreteModel()  # Model initialization


    l0 = rest_areas_0[col_highway].to_list()
    l1 = rest_areas_1[col_highway].to_list()

    for name in highway_names:
        if name in l0:
            dir0_extract_indices = rest_areas_0[rest_areas_0[col_highway] == name].index
            if len(dir0_extract_indices) > 0:
                rest_areas_0.loc[dir0_extract_indices[0], "first"] = True
                rest_areas_0.loc[dir0_extract_indices[-1], "last"] = True
        if name in l1:
            dir1_extract_indices = rest_areas_1[rest_areas_1[col_highway] == name].index
            if len(dir1_extract_indices) > 0:
                rest_areas_1.loc[dir1_extract_indices[-1], "first"] = True
                rest_areas_1.loc[dir1_extract_indices[0], "last"] = True

    m0 = len(rest_areas_0)
    m1 = len(rest_areas_1)
    m2 = len(rest_areas)
    model.IDX_0 = range(m0)
    model.IDX_1 = range(m1)
    model.IDX_2 = range(m2)
    model.XX = [2] * m2


    # these are not variables but constants
    model.pXi = Var(model.IDX_2, within=Binary)
    model.pYi_dir_0 = Var(model.IDX_0, within=Integers)
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    model.pE_input_0 = Var(model.IDX_0)
    model.pE_input_1 = Var(model.IDX_1)
    model.pE_output_0 = Var(model.IDX_0)
    model.pE_output_1 = Var(model.IDX_1)
    model.pE_charged_0 = Var(model.IDX_0)
    model.pE_charged_1 = Var(model.IDX_1)

    cars_per_day = hours_of_constant_charging / (acc / charging_capacity)
    energy = acc * cars_per_day  # (kWh) charging energy per day by one charging pole

    model.c1 = ConstraintList()

    firsts_0 = rest_areas_0[rest_areas_0["first"] == True].index
    for ind in firsts_0:
        model.c1.add(model.pE_input_0[ind] == 0)

    firsts_1 = rest_areas_1[rest_areas_1["first"] == True].index
    for ind in firsts_1:
        model.c1.add(model.pE_input_1[ind] == 0)


    model.c2 = ConstraintList()


    for name in highway_names:
        if name in l0:
            dir0_extract_indices = rest_areas_0[rest_areas_0[col_highway] == name].index
            if len(dir0_extract_indices) > 0:
                for kl in range(1, len(dir0_extract_indices)):
                    model.c2.add(model.pE_input_0[dir0_extract_indices[kl]] == model.pE_output_0[dir0_extract_indices[kl-1]])

        if name in l1:
            dir1_extract_indices = rest_areas_1[rest_areas_1[col_highway] == name].index
            if len(dir1_extract_indices) > 0:
                for kl in range(len(dir1_extract_indices)-2, -1, -1):
                    model.c2.add(model.pE_input_1[dir1_extract_indices[kl]] == model.pE_output_1[dir1_extract_indices[kl] +
                                                                                                 1])

    dir_names = rest_areas[col_rest_area_name].to_list()
    dir_directions = rest_areas[col_directions].to_list()
    dir_highways = rest_areas[col_highway].to_list()

    # lists with number of charging poles on stations
    num_cp_0 = rest_areas_0.cs_below_50kwh.to_list()
    num_cp_1 = rest_areas_1.cs_below_50kwh.to_list()

    #
    has_cs_0 = rest_areas_0[col_has_cs].to_list()
    has_cs_1 = rest_areas_1[col_has_cs].to_list()
    has_cs = rest_areas[col_has_cs].to_list()

    # establishing infrastructure
    model.c_existing_infrastructure = ConstraintList()
    model.c20 = ConstraintList()
    for ij in model.IDX_2:
        name = dir_names[ij]
        direc = dir_directions[ij]
        highway = dir_highways[ij]
        extract_ex_infr_0 = rest_areas_0[
            (
                    (rest_areas_0[col_rest_area_name] == name)
                    & (rest_areas_0[col_directions] == direc)
                    & (rest_areas_0[col_highway] == highway)
            )
        ].index.to_list()
        extract_ex_infr_1 = rest_areas_1[
            (
                    (rest_areas_1[col_rest_area_name] == name)
                    & (rest_areas_1[col_directions] == direc)
                    & (rest_areas_1[col_highway] == highway)
            )
        ].index.to_list()
        if len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0:
            ind_0 = extract_ex_infr_0[0]
            ind_1 = extract_ex_infr_1[0]
            model.c20.add((model.pYi_dir_0[model.IDX_0[ind_0]] + model.pYi_dir_1[model.IDX_1[ind_1]]) * energy >=
                          model.pE_charged_0[model.IDX_0[ind_0]] + model.pE_charged_1[model.IDX_1[ind_1]])

        elif len(extract_ex_infr_0) > 0:
            ind_0 = extract_ex_infr_0[0]
            model.c20.add((model.pYi_dir_0[model.IDX_0[ind_0]]) * energy >=
                          model.pE_charged_0[model.IDX_0[ind_0]])
        elif len(extract_ex_infr_1) > 0:
            ind_1 = extract_ex_infr_1[0]
            model.c20.add((model.pYi_dir_1[model.IDX_1[ind_1]]) * energy >=
                          model.pE_charged_1[model.IDX_1[ind_1]])

        if len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0 and has_cs[ij] == True:
            ind_0 = extract_ex_infr_0[0]
            ind_1 = extract_ex_infr_1[0]
            model.XX[ij] = 1
            model.c_existing_infrastructure.add(model.pXi[ij] == 1)
            model.c_existing_infrastructure.add(
                model.pYi_dir_0[model.IDX_0[ind_0]]
                + model.pYi_dir_1[model.IDX_1[ind_1]]
                == num_cp_0[model.IDX_0[ind_0]]
            )
            print(num_cp_0[model.IDX_0[ind_0]], name, ij)

        elif len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0 and not has_cs[ij] == True:
            ind_0 = extract_ex_infr_0[0]
            ind_1 = extract_ex_infr_1[0]
            model.XX[ij] = 0
            model.c_existing_infrastructure.add(model.pXi[ij] == 0)
            model.c_existing_infrastructure.add(
                model.pYi_dir_0[model.IDX_0[ind_0]]
                + model.pYi_dir_1[model.IDX_1[ind_1]]
                == 0
            )

        elif len(extract_ex_infr_0) > 0 and not len(extract_ex_infr_1) > 0 and has_cs[ij] == True:
            ind_0 = extract_ex_infr_0[0]
            model.c_existing_infrastructure.add(model.pXi[ij] == 1)
            model.XX[ij] = 1
            model.c_existing_infrastructure.add(model.pYi_dir_0[model.IDX_0[ind_0]] == num_cp_0[model.IDX_0[ind_0]])
            print(num_cp_0[model.IDX_0[ind_0]])

        elif len(extract_ex_infr_0) > 0 and not len(extract_ex_infr_1) > 0 and not has_cs[ij] == True:
            ind_0 = extract_ex_infr_0[0]
            model.XX[ij] = 0
            model.c_existing_infrastructure.add(model.pXi[ij] == 0)
            model.c_existing_infrastructure.add(model.pYi_dir_0[model.IDX_0[ind_0]] == 0)

        elif not len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0 and has_cs[ij] == True:
            ind_1 = extract_ex_infr_1[0]
            model.XX[ij] = 1
            model.c_existing_infrastructure.add(model.pXi[ij] == 1)
            model.c_existing_infrastructure.add(model.pYi_dir_1[model.IDX_1[ind_1]] == num_cp_1[model.IDX_1[ind_1]])
            print(num_cp_1[model.IDX_1[ind_1]])
        elif not len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0 and not has_cs[ij] == True:
            ind_1 = extract_ex_infr_1[0]
            model.XX[ij] = 0
            model.c_existing_infrastructure.add(model.pXi[ij] == 0)
            model.c_existing_infrastructure.add(model.pYi_dir_1[model.IDX_1[ind_1]] == 0)
        else:
            model.c_existing_infrastructure.add(model.pXi[ij] == 0)


    # zero-constraints
    model.c14 = ConstraintList()
    for ij in model.IDX_0:
        model.c14.add(model.pE_charged_0[ij] >= 0)
        model.c14.add(model.pE_input_0[ij] >= 0)
        model.c14.add(model.pE_output_0[ij] >= 0)
        model.c14.add(model.pYi_dir_0[ij] >= 0)

    for ij in model.IDX_1:
        model.c14.add(model.pE_charged_1[ij] >= 0)
        model.c14.add(model.pE_input_1[ij] >= 0)
        model.c14.add(model.pE_output_1[ij] >= 0)
        model.c14.add(model.pYi_dir_1[ij] >= 0)


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


    lasts_0 = rest_areas_0[rest_areas_0["last"]==True].index
    lasts_1 = rest_areas_1[rest_areas_1["last"]==True].index

    model.obj = Objective(expr=(sum(model.pE_output_0[ind] for ind in lasts_0) + sum(model.pE_output_1[ind] for ind in
                                                                                     lasts_1)), sense=minimize)


    opt = SolverFactory("gurobi")
    opt_success = opt.solve(model)
    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")

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

    print(model.obj.value())
    print(pE_charged_1)
    print(pE_charged_0)


    # saving remaining energy demand
    for name in highway_names:
        extract_rest_areas_0 = rest_areas_0[rest_areas_0.highway == name]
        inds = extract_rest_areas_0.index.to_list()
        remaining_charge = 0
        for ij in range(len(extract_rest_areas_0)-1, -1, -1):
            remaining_demand = energy_demand_0[ij] * eta * mu - pE_charged_0[ij] - remaining_charge
            if remaining_demand <= 0:
                rest_areas_0.at[inds[ij], 'remaining_demand'] = 0
            else:
                rest_areas_0.at[inds[ij], 'remaining_demand'] = remaining_demand
            if remaining_demand >= 0:
                remaining_charge = 0
            else:
                remaining_charge = np.absolute(remaining_demand)

        extract_rest_areas_1 = rest_areas_1[rest_areas_1.highway == name]
        inds = extract_rest_areas_1.index.to_list()
        remaining_charge = 0
        for ij in range(0, len(extract_rest_areas_1)):
            remaining_demand = energy_demand_1[ij] * eta * mu - pE_charged_1[ij] - remaining_charge
            if remaining_demand <= 0:
                rest_areas_1.at[inds[ij], 'remaining_demand'] = 0
            else:
                rest_areas_1.at[inds[ij], 'remaining_demand'] = remaining_demand
            if remaining_demand >= 0:
                remaining_charge = 0
            else:
                remaining_charge = np.absolute(remaining_demand)

    remaining_demand_0 = rest_areas_0.remaining_demand.to_list()
    remaining_demand_1 = rest_areas_1.remaining_demand.to_list()

    total_poles = []

    for ij in model.IDX_2:
        name = dir_names[ij]
        direc = dir_directions[ij]
        highway = dir_highways[ij]
        extract_ex_infr_0 = rest_areas_0[
            (
                    (rest_areas_0[col_rest_area_name] == name)
                    & (rest_areas_0[col_directions] == direc)
                    & (rest_areas_0[col_highway] == highway)
            )
        ].index.to_list()
        extract_ex_infr_1 = rest_areas_1[
            (
                    (rest_areas_1[col_rest_area_name] == name)
                    & (rest_areas_1[col_directions] == direc)
                    & (rest_areas_1[col_highway] == highway)
            )
        ].index.to_list()
        if len(extract_ex_infr_0) > 0 and len(extract_ex_infr_1) > 0:
            ind_0 = extract_ex_infr_0[0]
            ind_1 = extract_ex_infr_1[0]
            total_poles.append(pYi_dir_0[model.IDX_0[ind_0]] + pYi_dir_1[model.IDX_1[ind_1]])
        elif len(extract_ex_infr_0) > 0:
            ind_0 = extract_ex_infr_0[0]
            total_poles.append(pYi_dir_0[model.IDX_0[ind_0]])
        elif len(extract_ex_infr_1) > 0:
            ind_1 = extract_ex_infr_1[0]
            total_poles.append(pYi_dir_1[model.IDX_1[ind_1]])

    return remaining_demand_0, remaining_demand_1, pXi, total_poles