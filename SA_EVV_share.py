from optimization import *
from optimization_parameters import *
import matplotlib.pyplot as plt

epsilons = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

cs_nbs = []
poles_nbs = []
profits = []
for e in epsilons:
    nb_cs, nb_poles, profit = optimization(eta=e)
    cs_nbs.append(nb_cs)
    poles_nbs.append(nb_poles)
    profits.append(profit)

epsilons = [e * 100 for e in epsilons]
plt.figure()
plt.plot(epsilons, cs_nbs, "-")
plt.xlabel("EV share (%)")
plt.ylabel("number charging stations")
plt.savefig("SA/epsilon/epsilon_nb_cs.png")

plt.figure()
plt.plot(epsilons, poles_nbs, "-")
plt.xlabel("EV share (%)")
plt.ylabel("number of poles")
plt.savefig("SA/epsilon/epsilon_nb_poles.png")

plt.figure()
plt.plot(epsilons, profits, "-")
plt.xlabel("EV share (%)")
plt.ylabel("Yearly profit (€)")
plt.savefig("SA/epsilon/epsilon_tot_profit.png")

plt.figure()
plt.plot(epsilons, [profits[ij] / poles_nbs[ij] for ij in range(0, len(profits))], "-")
plt.xlabel("EV share (%)")
plt.ylabel("Yearly profit for one charging pole (€)")
plt.savefig("SA/epsilon/epsilon_pole_profit.png")
