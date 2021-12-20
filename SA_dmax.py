from optimization import *
from optimization_parameters import *
import matplotlib.pyplot as plt

dists = [10, 50, 100, 200, 300, 500, 1000]

cs_nbs = []
poles_nbs = []
profits = []
for d in dists:
    nb_cs, nb_poles, profit = optimization(maximum_dist_between_charging_stations=d)
    cs_nbs.append(nb_cs)
    poles_nbs.append(nb_poles)
    profits.append(profit)

plt.figure()
plt.plot(dists, cs_nbs, "-")
plt.xlabel("Maximum distance between charging stations (km)")
plt.ylabel("number charging stations")
plt.savefig("SA/dmax/dmax_nb_cs.png")

plt.figure()
plt.plot(dists, poles_nbs, "-")
plt.xlabel("Maximum distance between charging stations (km)")
plt.ylabel("number of poles")
plt.savefig("SA/dmax/dmax_nb_poles.png")

plt.figure()
plt.plot(dists, profits, "-")
plt.xlabel("Maximum distance between charging stations (km)")
plt.ylabel("Yearly profit (€)")
plt.savefig("SA/dmax/dmax_tot_profit.png")

plt.figure()
plt.plot(dists, [profits[ij] / poles_nbs[ij] for ij in range(0, len(profits))], "-")
plt.xlabel("Maximum distance between charging stations (km)")
plt.ylabel("Yearly profit for one charging pole (€)")
plt.savefig("SA/dmax/dmax_pole_profit.png")
