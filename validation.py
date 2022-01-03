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

# reading existing infrastructure

ex_infr_0 = pd.read_csv("data/rest_areas_0_charging_stations.csv")
ex_infr_1 = pd.read_csv("data/rest_areas_1_charging_stations.csv")

# join this with rest areas
rest_areas = pd2gpd(
    pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
).sort_values(by=["on_segment", "dist_along_highway"])
rest_areas["segment_id"] = rest_areas["on_segment"]
rest_areas[col_type_ID] = rest_areas["nb"]
rest_areas[col_directions] = rest_areas["evaluated_dir"]

rest_areas_0, rest_areas_1 = split_by_dir(rest_areas, col_directions, reindex=True)

existing_infr_0 = pd.merge(rest_areas_0, ex_infr_0, on=[col_highway, 'name', 'direction'])
existing_infr_1 = pd.merge(rest_areas_1, ex_infr_1, on=[col_highway, 'name', 'direction'])

# parameters status quo
cx = 7000
cy = 17750
eta = 0.01
average_driving_distance = 250  # (km)
long_dist = 100000  # (m)
dist_max = calculate_dist_max(average_driving_distance)
mu, gamma_h = calculate_max_util_params(long_dist)
charging_capacity = 50  # (kWh)
specific_demand = 25    # (kWh/100km)
acc = 50    # kW

# (1) ---------------------------------------------------------------------------------------------------------------

number_charging_poles_1, number_charging_stations_1, _, _ = optimization(
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
    mu,
    gamma_h,
    charging_capacity,
    specific_demand,
    introduce_existing_infrastructure_1,
    no_new_infrastructure_1,
    existing_infr_0,
    existing_infr_1,

)

# (2) ---------------------------------------------------------------------------------------------------------------

_, _, _, non_covered_energy = optimization(
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
    charging_capacity,
    specific_demand,
    introduce_existing_infrastructure_2,
    no_new_infrastructure_2,
    existing_infr_0,
    existing_infr_1,
)

# TODO: printing results of validation
