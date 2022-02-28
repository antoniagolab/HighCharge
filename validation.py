"""

This script is to validate the model by means of
    -> comparing the output to existing infrastructure

results of this validation script are saved to folder "validation_results/"

"""
from _parameter_calculations import *
from _optimization import *
from _file_import_optimization import *
from _utils import pd2gpd

no_new_infrastructure_1 = False
no_new_infrastructure_2 = True
introduce_existing_infrastructure_1 = False
introduce_existing_infrastructure_2 = True

# parameters status quo
cx = 40000
cy = 67000
eta = 0.015
average_driving_distance = 340  # (km)

long_dist = 100000  # (m)
dist_max = calculate_dist_max(average_driving_distance)
mu, gamma_h = calculate_max_util_params(long_dist)
charging_capacity = 81  # (kW)
specific_demand = 24  # (kWh/100km)
acc = charging_capacity # kW
a = 1
pole_peak_cap = 150  # (kW)
# (1) ---------------------------------------------------------------------------------------------------------------

number_charging_poles_1, number_charging_stations_1, _, _, _, = optimization(
    pois_df,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    cx,
    cy,
    dist_max,
    eta,
    acc,
    default_mu,
    default_gamma_h,
    a,
    charging_capacity,
    pole_peak_cap,
    specific_demand,
    introduce_existing_infrastructure_1,
    no_new_infrastructure_1,
    existing_infr_0,
    existing_infr_1,
    0.1,
    scenario_name="validation 1",
    path_res="validation_results/"
)

