"""

Python script calculating scenario parameters

    (1) eta: share of BEVs in Austrian car fleet
    (2) a: Changes in road passenger transport
    (3) specific_demand: (kWh/100km) specific energy demand of average BEV
    (4) dist_range: (km) driving range of average BEV


"""

import numpy as np
import matplotlib.pyplot as plt
from parameter_calculations import *

# (1) ---------------------------------------------------------------------------------------------------------------
# get previous development of BEV purchases + total new registrations
# source: https://www.beoe.at/statistik/;
# 2018 to 2021
new_reg_ev = [6757, 9242, 15972, 29955]

# values of new registrations over the last years
# source: https://de.statista.com/statistik/daten/studie/287733/umfrage/anzahl-der-monatlichen-pkw-neuzulassungen-in-
# oesterreich/
# 2018 to 2021
# downwards trend -> but held steady for this analysis
new_reg = [341068, 329363, 248740, 245671]
nb_new_reg = np.average(new_reg)

# for 2030: nb_new_reg = registrations BEVs
perc_ev_reg = (np.array(new_reg_ev) / np.array(new_reg)) * 100

x = list(range(2018, 2022)) + [2030]
y = list(perc_ev_reg) + [100]

# assuming linear growth of percentages of newly registered vehicles
# to calculate: k, d

k = (y[len(x)-1] - y[len(x)-2]) / (x[len(x)-1] - x[len(x)-2])
d = y[len(x)-1] - k * x[len(x)-1]


k2 = (y[len(x)-2] - y[len(x)-3]) / (x[len(x)-2] - x[len(x)-3])
d2 = y[len(x)-2] - k2 * x[len(x)-2]

x_pred = np.array(range(2022, 2030))
y_pred = k * x_pred + d

x_pred_2 = np.array(range(2022, 2031))
y_pred_2 = k2 * x_pred_2 + d2

 # plt.rcParams['font.sans-serif'] = ['Tahoma']

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'

plt.plot(x[:-1], y[:-1], 'o', label='past pre')
plt.plot(x_pred, y_pred, 'o', color='green', label='predicted new registrations of BEVs (optimal)')
plt.plot(x_pred_2, y_pred_2, 'o', color='red', label='predicted new registrations of BEVs (less optimal)')
plt.plot(x[len(x)-1], y[len(y)-1], 'o', color='orange', label='2040 goal')
plt.xlabel('year')
plt.ylabel('share of BEV new registrations')
plt.grid()
plt.legend()

# projection of the new registration onto car fleet
# overall number of passenger cars in Austrian car fleet
# source: https://de.statista.com/statistik/daten/studie/150173/umfrage/bestand-an-pkw-in-oesterreich/
# years 2018 to 2020
# from now on steady amount of cars is assumed due to ongoing modal shift and phase-out of combustion vehicles

registered_cars = [4898578, 5039548, 5091827]
future_of_total_registered_cars = np.average(registered_cars)

share_newly_reg = nb_new_reg/future_of_total_registered_cars    # yearly share of newly registered vehicles

# source: https://www.beoe.at/statistik/
# 2018 - 2021
registered_bevs_total_2021 = [20831, 29500, 44500, 80000]

registered_electric_vehicles = []
registered_electric_vehicles_2 = []

currently_registerd_bevs = registered_bevs_total_2021[-1]
currently_registerd_bevs_2 = currently_registerd_bevs

shares_of_add_bev_reg = list(y_pred) + [100]
shares_of_add_bev_reg_2 = list(y_pred_2)
for ij in range(0, len(shares_of_add_bev_reg)):
    currently_registerd_bevs = currently_registerd_bevs + shares_of_add_bev_reg[ij]/100 * nb_new_reg
    currently_registerd_bevs_2 = currently_registerd_bevs_2 + shares_of_add_bev_reg_2[ij]/100 * nb_new_reg
    registered_electric_vehicles.append(currently_registerd_bevs)
    registered_electric_vehicles_2.append(currently_registerd_bevs_2)


ev_shares = np.array(registered_bevs_total_2021 + registered_electric_vehicles)/future_of_total_registered_cars * 100
ev_shares_2 = np.array(registered_bevs_total_2021 + registered_electric_vehicles_2)/future_of_total_registered_cars * 100

# display of development of share of BEVs
fig, ax = plt.subplots()
plt.rcParams["font.family"] = "serif"
# plt.fill_between(list(range(2018, 2031)), [100] * len(list(range(2018, 2031))), label='registered passenger vehicles')
plt.fill_between(list(range(2018, 2031)), ev_shares,
                 label="optimal case: share of registered BEVs", color='green')
plt.fill_between(list(range(2018, 2031)), ev_shares_2,
                 label="less optimal case: share of registered BEVs", color='red')
plt.text(2030.5, ev_shares[-1], str(round(ev_shares[-1], 2)) + '%', multialignment='left', family="serif")
plt.text(2030.5, ev_shares_2[-1], str(round(ev_shares_2[-1], 2)) + '%', multialignment='left', family="serif")
plt.legend(loc='upper left')
plt.grid()
plt.xlabel('year')
plt.ylabel('share of vehicles in Austrian car fleet (%)')
plt.xlim(2018, 2032)


# (2) ---------------------------------------------------------------------------------------------------------------
# a ... share of ways
# according to AMMP: reduction from 61% of ways to 42% needed

total_red_until_2040 = 1 - 42/61
annual_reduction = total_red_until_2040/(2040-2018)

red_until_2030 = (2030-2018) * annual_reduction
road_transport_2030 = 1 - red_until_2030

# (3) ---------------------------------------------------------------------------------------------------------------
# specific demand


# (4) ---------------------------------------------------------------------------------------------------------------
# driving range
# source: https://www.kfz-betrieb.vogel.de/die-zehn-beliebtesten-autos-in-oesterreich-gal-993852/?p=1#gallerydetail
range_top_cars = [295, 190, 300, 235, 190, 190, 665, 680]
av_range_2020 = np.average(range_top_cars)

