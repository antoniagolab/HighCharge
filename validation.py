"""

This script is to validate the model by means of
    -> (1) comparing the output to existing infrastructure
    -> (2) assuming only existing infrastructure and determining the demand lack

"""
from parameter_calculations import *
from optimization import *
from file_import import *
from utils import pd2gpd

no_new_infrastructure_1 = False
no_new_infrastructure_2 = True
introduce_existing_infrastructure_1 = False
introduce_existing_infrastructure_2 = True

# parameters status quo
cx = 37000
cy = 60000
eta = 0.016
average_driving_distance = 555  # (km)
# average_driving_distance = 50  # (km)

long_dist = 100000  # (m)
dist_max = calculate_dist_max(average_driving_distance)
mu, gamma_h = calculate_max_util_params(long_dist)
charging_capacity = 80.45  # (kWh)
specific_demand = 23.87  # (kWh/100km)
acc = 41 * (4 / 6)  # kW
a = 1
# (1) ---------------------------------------------------------------------------------------------------------------
#
# number_charging_poles_1, number_charging_stations_1, _, _, _, = optimization(
#     pois_df,
#     dir_0,
#     dir_1,
#     segments_gdf,
#     links_gdf,
#     cx,
#     cy,
#     dist_max,
#     eta,
#     acc,
#     mu,
#     gamma_h,
#     a,
#     charging_capacity,
#     specific_demand,
#     introduce_existing_infrastructure_1,
#     no_new_infrastructure_1,
#     existing_infr_0,
#     existing_infr_1,
#     scenario_name="validation 1",
# )
#
# # (2) ---------------------------------------------------------------------------------------------------------------
#
# _, _, _, non_covered_energy, perc_not_charged = optimization(
#     dir,
#     dir_0,
#     dir_1,
#     segments_gdf,
#     links_gdf,
#     cx,
#     cy,
#     dist_max,
#     eta,
#     acc,
#     mu,
#     gamma_h,
#     a,
#     charging_capacity,
#     specific_demand,
#     introduce_existing_infrastructure_2,
#     no_new_infrastructure_2,
#     existing_infr_0,
#     existing_infr_1,
#     scenario_name="validation 2",
# )

# (3) ---------------------------------------------------------------------------------------------------------------

_, _, _, _, _ = optimization(
    dir,
    dir_0,
    dir_1,
    segments_gdf,
    links_gdf,
    cx,
    cy,
    dist_max,
    eta,
    acc,
    mu,
    gamma_h,
    a,
    charging_capacity,
    specific_demand,
    True,
    False,
    existing_infr_0,
    existing_infr_1,
    scenario_name="validation 3",
)


# TODO: printing results of validation

# existing infrastructure
existing_cs = 27
existing_cp = 160

print(
    "-----------------------------------------------------------------------------------"
)
print("(1) modelled infrastructure")
print(
    "existing:",
    str(existing_cs),
    " stations;",
    existing_cp,
    "poles;",
    "installed capacity:",
    installed_cap.sum(),
    "kW",
)
print(
    "estimated:",
    str(int(number_charging_poles_1)),
    "stations; ",
    int(number_charging_stations_1),
    "poles; installed capacity:",
    number_charging_poles_1 * 150, "kW"
)

print(
    "-----------------------------------------------------------------------------------"
)
print("(2) not covered demand")
print("not covered:", round(perc_not_charged, 2))
print(
    "-----------------------------------------------------------------------------------"
)
