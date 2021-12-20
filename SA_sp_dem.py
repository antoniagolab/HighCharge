from optimization import *
from optimization_parameters import *
import matplotlib.pyplot as plt
import numpy as np

consumptions = [0.2, 0.15, 0.1, 0.05, 0.025, 0.001]

cs_nbs = []
poles_nbs = []
profits = []
for c in consumptions:
    nb_cs, nb_poles, profit = optimization(specific_demand=c)
    cs_nbs.append(nb_cs)
    poles_nbs.append(nb_poles)
    profits.append(profit)

plt.figure()
plt.plot(consumptions, cs_nbs, '-')
plt.xlabel('energy consumption (kWh/km)')
plt.ylabel('number charging stations')
plt.savefig('SA/consumption/ec_nb_cs.png')

plt.figure()
plt.plot(consumptions, poles_nbs, '-')
plt.xlabel('energy consumption (kWh/km)')
plt.ylabel('number of poles')
plt.savefig('SA/consumption/ec_nb_poles.png')

plt.figure()
plt.plot(consumptions, profits, '-')
plt.xlabel('energy consumption (kWh/km)')
plt.ylabel('Yearly profit (€)')
plt.savefig('SA/consumption/ec_tot_profit.png')

plt.figure()
plt.plot(consumptions, [profits[ij]/poles_nbs[ij] for ij in range(0, len(profits))], '-')
plt.xlabel('energy consumption (kWh/km)')
plt.ylabel('Yearly profit for one charging pole (€)')
plt.savefig('SA/consumption/ec_pole_profit.png')
