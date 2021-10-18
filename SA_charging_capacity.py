from optimization import *
from optimization_parameters import *
import matplotlib.pyplot as plt

capacities = [50, 150, 250, 350]

cs_nbs = []
poles_nbs = []
profits = []
for c in capacities:
    nb_cs, nb_poles, profit = optimization(charging_capacity=c)
    cs_nbs.append(nb_cs)
    poles_nbs.append(nb_poles)
    profits.append(profit)

plt.figure()
plt.plot(capacities, cs_nbs, '-')
plt.xlabel('Charging capacity (kW)')
plt.ylabel('number charging stations')
plt.savefig('SA/charging_cap/charging_cap_nb_cs.png')

plt.figure()
plt.plot(capacities, poles_nbs, '-')
plt.xlabel('Charging capacity (kW)')
plt.ylabel('number of poles')
plt.savefig('SA/charging_cap/charging_cap_nb_poles.png')

plt.figure()
plt.plot(capacities, profits, '-')
plt.xlabel('Charging capacity (kW)')
plt.ylabel('Yearly profit (€)')
plt.savefig('SA/charging_cap/charging_cap_tot_profit.png')

plt.figure()
plt.plot(capacities, [profits[ij]/poles_nbs[ij] for ij in range(0, len(profits))], '-')
plt.xlabel('Charging capacity (kW)')
plt.ylabel('Yearly profit for one charging pole (€)')
plt.savefig('SA/charging_cap/charging_cap_pole_profit.png')
