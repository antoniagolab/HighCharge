"""

Sensitivity analysis (1)
    -> obeservance of


"""
import pandas as pd
from variable_definitions import *
from optimization import optimization
from parameter_calculations import *
from file_import import *
import datetime
import numpy as np


epsilon_max = 80
epsilon_min = 10

step_size = 10
r = 420
cy = 127000
eta = 34
charging_cap = 170
peak_pole = 350
output_file = pd.DataFrame()
a = 0.69

for eta in [100]:
# for eta in range(epsilon_min, epsilon_max + step_size, step_size):

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

output_file.to_csv("sensitivity_analyses/sensitivity_anal_epsilon_part_SC_0201_4.csv")
