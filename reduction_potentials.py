"""

Testing cost reduction potentials in Gradual Development scenario

"""
import pandas as pd
from variable_definitions import *
from optimization import optimization
from parameter_calculations import *
from file_import import *
import datetime

scenario_file = pd.read_csv('scenarios/scenario_parameters.csv')
scenario_file = scenario_file.set_index('scenario name')

name = 'Gradual Development'

new_epsilon = 0.34
new_a_0 = 0.84
new_a_1 = 0.69

range_2030_new = 1000
p_charge_2030_new = 400

p_charge_new = 245
range_new = 660

pole_peak_cap = 350
output_file = pd.DataFrame()


# trying new a -- 1
# scenario_name = 'e=34'
#
# nb_cs, nb_poles, costs, non_covered_energy, perc_not_charged = optimization(
#     dir,
#     dir_0,
#     dir_1,
#     segments_gdf,
#     links_gdf,
#     default_cx,
#     scenario_file.loc[name].cy,
#     calculate_dist_max(scenario_file.loc[name].dist_range/1000),
#     34/100,
#     default_acc,
#     default_mu,
#     default_gamma_h,
#     1,
#     scenario_file.loc[name].p_max_bev,
#     pole_peak_cap,
#     default_specific_demand,
#     True,
#     False,
#     existing_infr_0,
#     existing_infr_1,
#     0,
#     scenario_name
# )
# output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs, 'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy, 'perc_not_charged': perc_not_charged, 'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)



# trying new a -- 1
scenario_name = 'a=0.84'

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
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs, 'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy, 'perc_not_charged': perc_not_charged, 'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)


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
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs, 'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy, 'perc_not_charged': perc_not_charged, 'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)

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
output_file = output_file.append({'init_scenario': name, 'scenario_name': scenario_name, 'nb_cs': nb_cs, 'nb_poles': nb_poles, 'costs': costs, 'non_covered_energy': non_covered_energy, 'perc_not_charged': perc_not_charged, 'datetime_of_calculation': datetime.datetime.now()}, ignore_index=True)


output_file.to_csv('sensitivity_analyses/cost_reduction_potentials_2.csv')

