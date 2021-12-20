from optimization import *
from optimization_parameters import *
import matplotlib.pyplot as plt
import numpy as np

c_el = np.arange(0.2, 0.65, 0.05)

cs_nbs = []
poles_nbs = []
profits = []
for e in c_el:
    nb_cs, nb_poles, profit = optimization(ec=e)
    cs_nbs.append(nb_cs)
    poles_nbs.append(nb_poles)
    profits.append(profit)

plt.figure()
plt.plot(c_el, cs_nbs, "-")
plt.xlabel("Retail electricity price (€/kWh)")
plt.ylabel("number charging stations")
plt.savefig("SA/c_el/c_el_nb_cs.png")

plt.figure()
plt.plot(c_el, poles_nbs, "-")
plt.xlabel("Retail electricity price (€/kWh)")
plt.ylabel("number of poles")
plt.savefig("SA/c_el/c_el_nb_poles.png")

plt.figure()
plt.plot(c_el, profits, "-")
plt.xlabel("Retail electricity price (€/kWh)")
plt.ylabel("Yearly profit (€)")
plt.savefig("SA/c_el/c_el_tot_profit.png")

plt.figure()
plt.plot(c_el, [profits[ij] / poles_nbs[ij] for ij in range(0, len(profits))], "-")
plt.xlabel("Retail electricity price (€/kWh)")
plt.ylabel("Yearly profit for one charging pole (€)")
plt.savefig("SA/c_el/c_el_pole_profit.png")
