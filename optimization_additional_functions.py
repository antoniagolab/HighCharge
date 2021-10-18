from optimization_parameters import *


def profitable(r, ec, T, e_tax, cfix, cvar, demand, energy):
    is_profitable = False
    number_poles = int(energy/demand)
    profit = ((-(cfix + cvar*number_poles) + sum([demand*(ec-e_tax) * 365/((1 + r)**n ) for n in range(1, T)])))
    print((-(cfix + cvar*number_poles) + sum([demand*(ec-e_tax) * 365/((1 + r)**n ) for n in range(1, T)])))
    if profit > 0:
        is_profitable = True
        num_var = 1
        if ((-(cfix + cvar*(number_poles+1)) + sum([(energy* (number_poles + 1))*(ec-e_tax) * 365/((1 + r)**n ) for n in range(1, T)]))) > 0:
            return is_profitable, number_poles+1
        else:
            return is_profitable, number_poles
    else:
        return is_profitable, 0


