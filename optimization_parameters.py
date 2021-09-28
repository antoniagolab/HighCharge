"""

Definition of parameters for charging station allocation

"""
import numpy as np
import pandas as pd

dir_0 = pd.read_csv(
    "data/rest_area_0_input_optimization_v3.csv"
)  # file containing service stations for both directions + for "normal" direction

dir_1 = pd.read_csv(
    "data/rest_area_1_input_optimization_v3.csv"
)  # file containing service stations for both directions + for "inverse" direction

col_energy_demand = (
    "energy_demand"  # specification of column name for energy demand per day
)
col_directions = "direction"  # specification of column name for directions: 0 = 'normal'; 1 = 'inverse'; 2 = both directions
col_rest_area_name = "name"  # specification of column nameholding names of rest areas
col_traffic_flow = "traffic_flow"
col_type = "asfinag_type"
col_position = "asfinag_position"
col_highway = "highway"

dir = dir_0.append(dir_1)
dir = dir[[col_highway, col_rest_area_name, col_position, col_directions]]
dir = dir.sort_values(by=[col_highway, col_position])
dir = dir.drop_duplicates(subset=[col_rest_area_name, col_directions])
dir["ID"] = range(0, len(dir))
dir = dir.set_index("ID")

n0 = len(dir_0)
n1 = len(dir_1)
k = len(dir)
n3 = k
# 50 kW annehmen
g = 10000  # Maximum number of charging poles at one charging station
specific_demand = 0.2  # (kWh/100km) average specific energy usage for 100km
acc = (
    specific_demand * 100
)  # (kWh) charged energy by a car during one charging for 100km
charging_capacity = 50  # (kW)
ec = 0.25  # (€/kWh) charging price for EV driver
e_tax = 0.15  # (€/kWh) total taxes and other charges
cfix = 50000  # (€) total installation costs of charging station installation
cvar1 = 10000  # (€) total installation costs of charging pole installation
eta = 0.011  # share of electric vehicles of car fleet
mu = 0.5    # share of vehicles which travel long-distance and might have to charge along highway
tf_at_peak = 0.14    # share of accumulation at peak demand time (hourly res.) of all cars passed throughout the day
hours_of_constant_charging = (
    20  # number of hours of continuous charging at one charging pole
)
hourly_res = 1  # (h)
energy_demand_0 = [d * tf_at_peak for d in dir_0[
    col_energy_demand].to_list()]   # (kWh/d) energy demand at each rest area per day
energy_demand_1 = [d * tf_at_peak for d in dir_1[col_energy_demand].to_list()]
directions_0 = dir_0[col_directions].to_list()
directions_1 = dir_1[col_directions].to_list()

cars_per_hour = hourly_res / (acc / charging_capacity)

time_of_charging = acc / charging_capacity
max_charging_at_pp = 10/60  # (h) maximum time of charging at a parking place


energy = acc * cars_per_hour  # (kWh) charging energy in resolution time t by one charging pole

e_average = (
    (sum(energy_demand_0) + sum(energy_demand_1)) * tf_at_peak * mu * eta / (n0 + n1)
)  # (kWh/d) average energy demand at a charging station per day
i = 0.05  # interest rate
T = 10  # (a) calculation period für RBF (=annuity value)
RBF = 1 / i - 1 / (i * (1 + i) ** T)  # (€/a) annuity value for period T

maximum_dist_between_charging_stations = 100  # (km)
traffic_flows_dir_0 = dir_0[col_traffic_flow].to_list()
traffic_flows_dir_1 = dir_1[col_traffic_flow].to_list()

# extracting all highway names to create two additional columns: "first" and "last" to indicate whether resting areas
# are first resting areas along a singular highway in "normal" direction
dir_0["first"] = [False] * n0
dir_0["last"] = dir_0["first"]
dir_1["first"] = [False] * n1
dir_1["last"] = dir_1["first"]
l1 = dir_1[col_highway].to_list()
l0 = dir_0[col_highway].to_list()
l_ext = l0
l_ext.extend(l1)
highway_names = list(set(l_ext))
for name in highway_names:
    if name in l0:
        dir0_extract_indices = dir_0[dir_0[col_highway] == name].index
        if len(dir0_extract_indices) > 0:
            dir_0.loc[dir0_extract_indices[0], "first"] = True
            dir_0.loc[dir0_extract_indices[-1], "last"] = True
    if name in l1:
        dir1_extract_indices = dir_1[dir_1[col_highway] == name].index
        if len(dir1_extract_indices) > 0:
            dir_1.loc[dir1_extract_indices[0], "first"] = True
            dir_1.loc[dir1_extract_indices[-1], "last"] = True

e_average_0 = {}
e_average_1 = {}

for name in highway_names:
    energy_demand_dir_0 = dir_0[dir_0[col_highway] == name][col_energy_demand].to_list()
    if len(energy_demand_dir_0) > 0:
        e_average_0[name] = np.average(energy_demand_dir_0) * tf_at_peak * eta * mu
    else:
        e_average_0[name] = 0

    energy_demand_dir_1 = dir_1[dir_1[col_highway] == name][col_energy_demand].to_list()
    if len(energy_demand_dir_1) > 0:
        e_average_1[name] = np.average(energy_demand_dir_1) * tf_at_peak * eta * mu
    else:
        e_average_1[name] = 0


# TODO ADDING PARAMETERS:
#   - traffic density to rush hour (check)
#   - percentage of cars on highway which are travelling long-distance (and therefore need to charge on highway) (check)


