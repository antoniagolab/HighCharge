"""

Python script for all analysis

"""

import pandas as pd
from _variable_definitions import *
from _optimization import optimization
from _parameter_calculations import *
from _file_import_optimization import *
import datetime

# ---------------------------------------------------------------------------------------------------------------------
# SCENARIO calculation
# ---------------------------------------------------------------------------------------------------------------------


scenario_file = pd.read_csv("scenarios/scenario_parameters.csv")

l = len(scenario_file)
names = scenario_file["scenario name"].to_list()
etas = scenario_file["eta"].to_list()
mus = scenario_file["mu"].to_list()
gamma_hs = scenario_file["gamma_h"].to_list()
a_s = scenario_file["a"].to_list()
specific_demands = scenario_file["specific_demand"]
cxs = scenario_file["cx"]
cys = scenario_file["cy"]
dist_ranges = scenario_file["dist_range"]
p_max_bevs = scenario_file["p_max_bev"]
pole_peak_cap = 350
output_file = pd.DataFrame()
existing_infr["installed_infrastructure"] = existing_infr["350kW"]
existing_infr_0, existing_infr_1 = split_by_dir(existing_infr, "dir", reindex=True)

for ij in range(0, l-2):
    scenario_name = names[ij]
    if not etas[ij] >= 0:
        eta = default_eta
    else:
        eta = etas[ij]

    if not cxs[ij] >= 0:
        cx = default_cx
    else:
        cx = cxs[ij]

    if not cys[ij] >= 0:
        cy = default_cy
    else:
        cy = cys[ij]

    if not dist_ranges[ij] >= 0:
        dist_range = default_dmax
    else:
        dist_range = dist_ranges[ij]

    if not mus[ij] >= 0:
        mu = default_mu
    else:
        mu = mus[ij]

    if not gamma_hs[ij] >= 0:
        gamma_h = default_gamma_h
    else:
        gamma_h = gamma_hs[ij]

    if not a_s[ij] >= 0:
        a = default_a
    else:
        a = a_s[ij]

    if not p_max_bevs[ij] >= 0:
        p_max_bev = default_charging_capacity
    else:
        p_max_bev = p_max_bevs[ij]

    if not specific_demands[ij] >= 0:
        specific_demand = default_specific_demand
    else:
        specific_demand = specific_demands[ij]

    nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
        dir,
        dir_0,
        dir_1,
        segments_gdf,
        links_gdf,
        cx,
        cy,
        calculate_dist_max(dist_range),
        eta / 100,
        default_acc,
        mu,
        gamma_h,
        a,
        p_max_bev,
        pole_peak_cap,
        specific_demand,
        True,
        False,
        existing_infr_0,
        existing_infr_1,
        0,
        scenario_name,
        path_res="scenarios/"
    )
    output_file = output_file.append(
        {
            "nb_cs": nb_cs,
            "nb_poles": nb_poles,
            "costs": costs,
            "non_covered_energy": non_covered_energy,
            "perc_not_charged": perc_not_charged,
            "datetime_of_calculation": datetime.datetime.now(),
        },
        ignore_index=True,
    )


output_file.to_csv("scenarios/scenario_results.csv")


# ---------------------------------------------------------------------------------------------------------------------
# REDUCTION POTENTIALS
# ---------------------------------------------------------------------------------------------------------------------

existing_infr["installed_infrastructure"] = existing_infr["350kW"]
existing_infr_0, existing_infr_1 = split_by_dir(existing_infr, "dir", reindex=True)
scenario_file = pd.read_csv('scenarios/scenario_parameters.csv')
scenario_file = scenario_file.set_index('scenario name')

name = 'Gradual Development'

new_epsilon = 0.33
new_a_0 = 0.83
new_a_1 = 0.69

range_2030_new = 1000
p_charge_2030_new = 400

p_charge_new = 242.5
range_new = 660

pole_peak_cap = 350
output_file = pd.DataFrame()


# trying new a -- 1
scenario_name = 'a=0.83'

nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    default_cx,
    scenario_file.loc[name].cy,
    calculate_dist_max(scenario_file.loc[name].dist_range),
    scenario_file.loc[name].eta/100,
    default_acc,
    default_mu,
    default_gamma_h,
    new_a_0,
    scenario_file.loc[name].p_max_bev,
    pole_peak_cap,
    default_specific_demand,
    True,
    False,
    existing_infr_0,
    existing_infr_1,
    0,
    scenario_name
)
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs,
                                  'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy,
                                  'perc_not_charged': perc_not_charged,
                                  'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)


# trying new a -- 2
scenario_name = 'a=0.69'
nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    default_cx,
    scenario_file.loc[name].cy,
    calculate_dist_max(scenario_file.loc[name].dist_range),
    scenario_file.loc[name].eta/100,
    default_acc,
    default_mu,
    default_gamma_h,
    new_a_1,
    scenario_file.loc[name].p_max_bev,
    pole_peak_cap,
    default_specific_demand,
    True,
    False,
    existing_infr_0,
    existing_infr_1,
    0,
    scenario_name
)
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs,
                                  'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy,
                                  'perc_not_charged': perc_not_charged,
                                  'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)

# trying new range

scenario_name = 'range=' + str(range_new)
nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    default_cx,
    scenario_file.loc[name].cy,
    calculate_dist_max(range_new),
    scenario_file.loc[name].eta/100,
    default_acc,
    default_mu,
    default_gamma_h,
    scenario_file.loc[name].a,
    scenario_file.loc[name].p_max_bev,
    pole_peak_cap,
    default_specific_demand,
    True,
    False,
    existing_infr_0,
    existing_infr_1,
    0,
    scenario_name
)
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs, 'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy, 'perc_not_charged': perc_not_charged, 'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)

# trying new range

scenario_name = '1=' + str(range_new)
nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    default_cx,
    scenario_file.loc[name].cy,
    calculate_dist_max(scenario_file.loc[name].dist_range),
    scenario_file.loc[name].eta/100,
    default_acc,
    default_mu,
    default_gamma_h,
    scenario_file.loc[name].a,
    p_charge_new,
    pole_peak_cap,
    default_specific_demand,
    True,
    False,
    existing_infr_0,
    existing_infr_1,
    0,
    scenario_name
)
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs,
                                  'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy,
                                  'perc_not_charged': perc_not_charged,
                                  'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)


output_file.to_csv('sensitivity_analyses/cost_reduction_potentials.csv')


# ---------------------------------------------------------------------------------------------------------------------
# SENSITIVITY ANALYSIS I: driving range
# ---------------------------------------------------------------------------------------------------------------------
range_max = 1400
range_min = 200

step_size = 100
cx = 300000
cy = 127000
eta = 33
a = 0.83
charging_cap = 247.8
peak_pole = 350
output_file = pd.DataFrame()


for r in range(range_min, range_max + step_size, step_size):
    scen_name = "MIPFOCUS=0 driving range TF _2" + str(int(r)) + " km - eta 2" + str(eta)

    nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
        dir,
        dir_0,
        dir_1,
        segments_gdf,
        links_gdf,
        cx,
        cy,
        r * (4/5) * 0.6 * 1000,
        eta / 100,
        default_acc,
        default_mu,
        default_gamma_h,
        a,
        charging_cap,
        peak_pole,
        default_specific_demand,
        False,
        False,
        existing_infr_0,
        existing_infr_1,
        0,
        scenario_name="TF" + str(r),
        path_res="sensitivity_analyses/driving_range/"
    )
    output_file = output_file.append(
        {
            "scenario": "TC",
            "dist_range": r,
            "nb_cs": nb_cs,
            "nb_poles": nb_poles,
            "costs": costs,
            "non_covered_energy": non_covered_energy,
            "perc_not_charged": perc_not_charged,
            "datetime_of_calculation": datetime.datetime.now(),
        },
        ignore_index=True,
    )

output_file.to_csv(
    "sensitivity_analyses/driving_range&sensitivity_anal_TF.csv"
)


# ---------------------------------------------------------------------------------------------------------------------
# SENSITIVITY ANALYSIS II: share of BEVs in car fleet
# ---------------------------------------------------------------------------------------------------------------------

epsilon_max = 70
epsilon_min = 10

step_size = 10
r = 420
cy = 127000
eta = 33
charging_cap = 165.8
peak_pole = 350
output_file = pd.DataFrame()
a = 0.69

for eta in range(epsilon_min, epsilon_max + step_size, step_size):

    scen_name = "ev share - epsilon SC" + str(eta) + "_3"

    nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
        dir,
        dir_0,
        dir_1,
        segments_gdf,
        links_gdf,
        default_cx,
        cy,
        calculate_dist_max(r),
        eta / 100,
        default_acc,
        default_mu,
        default_gamma_h,
        default_a,
        charging_cap,
        peak_pole,
        default_specific_demand,
        False,
        False,
        existing_infr_0,
        existing_infr_1,
        0,
        scenario_name=scen_name,
        path_res="sensitivity_analyses/bev_share/"
    )
    output_file = output_file.append(
        {
            "epsilon": eta,
            "nb_cs": nb_cs,
            "nb_poles": nb_poles,
            "costs": costs,
            "non_covered_energy": non_covered_energy,
            "perc_not_charged": perc_not_charged,
            "datetime_of_calculation": datetime.datetime.now(),
        },
        ignore_index=True,
    )

output_file.to_csv("sensitivity_analyses/bev_share/sensitivity_anal_SC.csv")



