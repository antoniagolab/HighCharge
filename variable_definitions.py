
col_energy_demand = (
    "energy_demand"  # specification of column name for energy demand per day
)
col_directions = "dir"  # specification of column name for directions: 0 = 'normal'; 1 = 'inverse'; 2 = both directions
col_rest_area_name = "name"  # specification of column name holding names of rest areas
col_traffic_flow = "traffic_flow"
col_type = "pois_type"
# col_position = "asfinag_position"
col_highway = "highway"
col_has_cs = "has_charging_station"
col_distance = "dist_along_segment"
col_segment_id = "segment_id"
col_type_ID = "type_ID"
col_POI_ID = "POI_ID"

# default input values

default_specific_demand = 25  # (kWh/100km) average specific energy usage for 100km

default_acc = (
    default_specific_demand * 100
)  # (kWh) charged energy by a car during one charging for 100km
default_charging_capacity = 110  # (kW)
default_energy = default_charging_capacity  # (kWh/h)
default_ec = 0.25  # (€/kWh) charging price for EV driver
default_e_tax = 0.15  # (€/kWh) total taxes and other charges
default_cx = 7000  # (€) total installation costs of charging station installation
default_cy = 17750  # (€) total installation costs of charging pole installation
default_eta = 0.33  # share of electric vehicles of car fleet
default_mu = 0.18  # share of cars travelling long-distance
default_gamma_h = 0.125   # share of cars travelling during peak hour
default_a = 0.69

# dmax = 50000
default_dmax = 50000
