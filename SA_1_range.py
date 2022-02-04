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


range_max = 1100
range_min = 200

step_size = 100

cy = 127000
eta = 34
a = 0.83
charging_cap = 250
peak_pole = 350
output_file = pd.DataFrame()

# for r in [1400]:

for r in range(range_min, range_max + step_size, step_size):
    scen_name = "MIPFOCUS=0 driving range TF _2" + str(int(r)) + " km - eta 2" + str(eta)

    nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
        dir,
        dir_0,
        dir_1,
        segments_gdf,
        links_gdf,
        default_cx,
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
    "sensitivity_analyses/sensitivity_anal_TF_" + str(eta) + "_new_0203.csv"
)
