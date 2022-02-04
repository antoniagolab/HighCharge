"""

Python script for running all scenarios

"""

import pandas as pd
from variable_definitions import *
from optimization import optimization
from parameter_calculations import *
from file_import import *
import datetime

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

for ij in range(0, l):
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


output_file.to_csv("scenarios/optimization_results_new_3.csv")
