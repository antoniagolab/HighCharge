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

k = (y[len(x) - 1] - y[len(x) - 2]) / (x[len(x) - 1] - x[len(x) - 2])
d = y[len(x) - 1] - k * x[len(x) - 1]


k2 = (y[len(x) - 2] - y[len(x) - 3]) / (x[len(x) - 2] - x[len(x) - 3])
d2 = y[len(x) - 2] - k2 * x[len(x) - 2]

x_pred = np.array(range(2022, 2030))
y_pred = k * x_pred + d

x_pred_2 = np.array(range(2022, 2031))
y_pred_2 = k2 * x_pred_2 + d2


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 16
ax1.plot(x[:-1], y[:-1], "o", label="Past new registrations")
ax1.plot(
    x_pred,
    y_pred,
    "o",
    color="green",
    label="Optimal case: Predicted new registrations of BEVs",
)
ax1.plot(
    x_pred_2,
    y_pred_2,
    "o",
    color="red",
    label="Less optimal case: Predicted new registrations of BEVs",
)
ax1.plot(x[len(x) - 1], y[len(y) - 1], "o", color="orange", label="AMMP 2030 goal")
ax1.set_xlabel("year", fontsize=12)
ax1.set_ylabel("share of BEV new registrations (%)", fontsize=12)
ax1.set_ylim([0, 105])
ax1.set_title("Projection of new registration of BEVs", fontsize=13)
ax1.grid()
ax1.set_xticks(np.arange(2018, 2031, 2))
ax1.legend(loc="upper left", bbox_to_anchor=(-0.1, -0.15), fontsize=12)
plt.legend()


# projection of the new registration onto car fleet
# overall number of passenger cars in Austrian car fleet
# source: https://de.statista.com/statistik/daten/studie/150173/umfrage/bestand-an-pkw-in-oesterreich/
# years 2018 to 2020
# from now on steady amount of cars is assumed due to ongoing modal shift and phase-out of combustion vehicles

registered_cars = [4898578, 5039548, 5091827]
future_of_total_registered_cars = np.average(registered_cars)

share_newly_reg = (
    nb_new_reg / future_of_total_registered_cars
)  # yearly share of newly registered vehicles

# source: https://www.beoe.at/statistik/
# 2018 - 2021
registered_bevs_total_2021 = [20831, 29500, 44500, 76540]

registered_electric_vehicles = []
registered_electric_vehicles_2 = []

currently_registerd_bevs = registered_bevs_total_2021[-1]
currently_registerd_bevs_2 = currently_registerd_bevs

shares_of_add_bev_reg = list(y_pred) + [100]
shares_of_add_bev_reg_2 = list(y_pred_2)
for ij in range(0, len(shares_of_add_bev_reg)):
    currently_registerd_bevs = (
        currently_registerd_bevs + shares_of_add_bev_reg[ij] / 100 * nb_new_reg
    )
    currently_registerd_bevs_2 = (
        currently_registerd_bevs_2 + shares_of_add_bev_reg_2[ij] / 100 * nb_new_reg
    )
    registered_electric_vehicles.append(currently_registerd_bevs)
    registered_electric_vehicles_2.append(currently_registerd_bevs_2)


ev_shares = (
    np.array(registered_bevs_total_2021 + registered_electric_vehicles)
    / future_of_total_registered_cars
    * 100
)
ev_shares_2 = (
    np.array(registered_bevs_total_2021 + registered_electric_vehicles_2)
    / future_of_total_registered_cars
    * 100
)

# display of development of share of BEVs
# fig, ax = plt.subplots(figsize=(10, 5))
# plt.fill_between(list(range(2018, 2031)), [100] * len(list(range(2018, 2031))), label='registered passenger vehicles')
ax2.fill_between(
    list(range(2018, 2031)),
    ev_shares,
    label="Optimal case: share of registered BEVs",
    color="green",
)
ax2.fill_between(
    list(range(2018, 2031)),
    ev_shares_2,
    label="Less optimal case: share of registered BEVs",
    color="red",
)
ax2.text(
    2030.5,
    ev_shares[-1],
    str(int(round(round(ev_shares[-1],1),0))) + "%",
    multialignment="left", fontsize=12
)
ax2.text(
    2030.5,
    ev_shares_2[-1],
    str(int(round(ev_shares_2[-1], 0))) + "%",
    multialignment="left", fontsize=12
)
ax2.set_xticks(np.arange(2018, 2031, 2))
ax2.grid()
ax2.legend(loc="upper left", fontsize=12, bbox_to_anchor=(-0.01, -0.15))
ax2.set_xlabel("year", fontsize=12)
ax2.set_ylabel("share of BEVs in car fleet (%)", fontsize=12)
ax2.set_xlim(2018, 2032)
ax2.set_ylim(0, 37)
ax2.set_title("Projected development of share of BEVs", fontsize=13)
fig.tight_layout()
# plt.show()
plt.savefig('figures/appendix_img_meth.pdf')

# (2) ---------------------------------------------------------------------------------------------------------------
# a ... share of ways
# according to AMMP: reduction from 61% of ways to 42% needed

total_red_until_2040 = 1 - 42 / 61
annual_reduction = total_red_until_2040 / (2040 - 2018)

red_until_2030 = (2030 - 2018) * annual_reduction
road_transport_2030 = 1 - red_until_2030

# (5) ---------------------------------------------------------------------------------------------------------------
# charging capacity
# https://www.kleinezeitung.at/auto/elektroauto/5479610/Zulassungen-Jaenner-bis-November2021_Das-sind-die-meistverkauften#image-Tesla-Model_3-2018-1600-06_155290619725024_v0_h
maximum_capacities_today = [120, 64, 30, 216, 67, 41, 103, 103, 85, 110]
av_cap = 80.45

p_av_goal_opt = 315
p_av_goal_pess = 200


kc1 = (p_av_goal_opt - av_cap) / (2030 - 2020)
dc1 = p_av_goal_opt - kc1 * 2030

kc2 = (p_av_goal_pess - av_cap) / (2030 - 2020)
dc2 = p_av_goal_pess - kc2 * 2030

optimistic_estimation = np.array(range(2020, 2031)) * kc1 + dc1
pessimistic_estimation = np.array(range(2020, 2031)) * kc2 + dc2

av_charg_cap_opt_opt = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred))
av_charg_cap_opt_pess = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_2[0:-1]))
av_charg_cap_pess_pess = np.average(pessimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_2[0:-1]))
av_charg_cap_pess_opt = np.average(pessimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred))


print(av_charg_cap_opt_opt)
# (4) ---------------------------------------------------------------------------------------------------------------
# driving range
# source: https://www.kfz-betrieb.vogel.de/die-zehn-beliebtesten-autos-in-oesterreich-gal-993852/?p=1#gallerydetail
range_top_cars = [595, 455, 565, 355, 615, 600, 520, 795]
av_range_2021 = 337
optimistic_goal = 800
mediocre_goal = 600
pessimistic_goal = 450

kr1 = (optimistic_goal - av_range_2021) / (2030 - 2020)
dr1 = optimistic_goal - kr1 * 2030

kr2 = (pessimistic_goal - av_range_2021) / (2030 - 2020)
dr2 = pessimistic_goal - kr2 * 2030

kr3 = (mediocre_goal - av_range_2021) / (2030 - 2020)
dr3 = mediocre_goal - kr3 * 2030

optimistic_estimation = np.array(range(2020, 2031)) * kr1 + dr1
pessimistic_estimation = np.array(range(2020, 2031)) * kr2 + dr2
med_estimation = np.array(range(2020, 2031)) * kr3 + dr3

av_range_opt_opt = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred))
av_range_opt_pess = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_2[0:-1]))
av_range_pess_pess = np.average(med_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_2[0:-1]))
av_range_pess_opt = np.average(pessimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred))

y = 2040
optimistic_estimation = np.array(range(2020, y)) * kr1 + dr1
pessimistic_estimation = np.array(range(2020, y)) * kr2 + dr2
med_estimation = np.array(range(2020, y)) * kr3 + dr3

x_pred_00 = np.array(range(2022, y))
y_pred_00 = k * x_pred_00 + d

x_pred_01 = np.array(range(2022, y))
y_pred_01 = k2 * x_pred_01 + d2


#
#
# av_range_opt_opt = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_00[0:-1]))
# av_range_opt_pess = np.average(optimistic_estimation, weights=list(perc_ev_reg[1:]) + list(x_pred_01[0:-1]))
# av_range_pess_pess = np.average(med_estimation, weights=list(perc_ev_reg[1:]) + list(x_pred_01[0:-1]))
# av_range_pess_opt = np.average(pessimistic_estimation, weights=list(perc_ev_reg[1:]) + list(y_pred_00[0:-1]))
#

# calculate range for 2040: based on the last 10 years (so 2030 - 2040)
# constant 100% newregistrations; same technological improv rate; just averaged -> ergo: what is driving range 2035

BEVs_year_2040 = nb_new_reg * 0.69 * 10
cars_2040 = 5.1e6 * 0.69
bev_share = BEVs_year_2040/cars_2040