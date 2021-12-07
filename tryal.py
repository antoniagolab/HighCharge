
import pandas as pd
import numpy as np
from pyomo.environ import *
from optimization_parameters import *
from optimization_additional_functions import *
from integrating_exxisting_infrastructure import *
import time
import warnings
from pandas.core.common import SettingWithCopyWarning
from visualize_results import *
from termcolor import colored

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# parameter definition
energy = 40
demand = [20, 50, 60, 30, 20, 80, 5, 40]
energy_demand_matrix = np.diag(demand)
n = len(demand)

enum = np.array([[1, 2, 3, 4, 5, 6, 7, 0], [0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 2, 3, 4, 5]
                 , [0, 0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1]])


mask = np.where(enum > 0, 1, 0)


model = ConcreteModel()
model.idx = range(n)

# variables
model.charge = Var(model.idx, model.idx)
model.input = Var(model.idx, model.idx)
model.output = Var(model.idx, model.idx)

model.charge_tot = Var(model.idx)
model.input_tot = Var(model.idx)
model.output_tot = Var(model.idx)

model.X = Var(model.idx, within=Binary)
model.Y = Var(model.idx, within=Integers)
#
model.c = ConstraintList()
for ij in model.idx:    # check
    max_n = max(enum[ij, ])
    print(max_n)
    for kl in model.idx:
        if mask[ij, kl] == 0:
            model.c.add(model.charge[ij, kl] == 0)
            model.c.add(model.input[ij, kl] == 0)
            model.c.add(model.output[ij, kl] == 0)
        if enum[ij, kl] == 1:
            model.c.add(model.input[ij, kl] == 0)
        if enum[ij, kl] == max_n:
            model.c.add(model.output[ij, kl] == 0)

# model.c.add(model.charge[1,0] == 0)
# model.c.add(model.input[0,0]==0)
# model.c.add(model.input[1,0]==0)
# model.c.add(model.output[0,3]==0)
# model.c.add(model.output[1,3]==0)
# model.c.add(model.charge[2,0] == 0)
# model.c.add(model.charge[2,1] == 0)
# model.c.add(model.input[1,1]==0)
# model.c.add(model.output[3,3]==0)
# model.c.add(model.output[2,3]==0)


#
# model.c.add(model.charge_tot[0] == model.charge[0,0] + model.charge[1,0])
# model.c.add(model.charge_tot[1] == model.charge[0,1] + model.charge[1,1])
# model.c.add(model.input_tot[0] == model.input[0,0] + model.input[1,0])
# model.c.add(model.input_tot[1] == model.input[0,1] + model.input[1,1])
# model.c.add(model.output_tot[0] == model.output[0,0] + model.output[1,0])
# model.c.add(model.output_tot[1] == model.output[0,1] + model.output[1,1])

# changed sth
for ij in model.idx:
    for kl in model.idx:    # check
        model.c.add(model.input[ij, kl] - model.output[ij, kl] + energy_demand_matrix[ij, kl] - model.charge[ij, kl] == 0)
# # #
# model.c.add(model.input[0,0] - model.output[0,0] + energy_demand_matrix[0,0]-model.charge[0,0] == 0)
# model.c.add(model.input[0,1] - model.output[0,1] + energy_demand_matrix[0,1]-model.charge[0,1] == 0)
# model.c.add(model.input[1,0] - model.output[1,0] + energy_demand_matrix[1,0]-model.charge[1,0] == 0)
# model.c.add(model.input[1,1] - model.output[1,1] + energy_demand_matrix[1,1]-model.charge[1,1] == 0)

for ij in model.idx:    # check
    model.c.add(sum(model.input[kl, ij] for kl in model.idx) - sum(model.output[kl, ij] for kl in model.idx) -
               sum(model.charge[kl, ij] for kl in model.idx) + energy_demand_matrix[ij, ij] == 0)
# model.c.add((model.input[0,0] + model.input[1,0]) - (model.output[0,0] + model.output[1,0]) -
#                (model.charge[0,0] + model.charge[1,0]) + energy_demand_matrix[0, 0] == 0)
# model.c.add((model.input[0,1] + model.input[1,1]) - (model.output[0,1] + model.output[1,1]) -
#               (model.charge[0,1] + model.charge[1,1]) + energy_demand_matrix[1, 1] == 0)

for ij in range(0, n):  #check
    current_enum = enum[ij, ]
    ind_sortation_list = np.argsort(current_enum)
    ind_sortation_list = [kl for kl in ind_sortation_list if not current_enum[kl] == 0]
    for mn in range(1, len(ind_sortation_list)):
        model.c.add(model.output[ij, ind_sortation_list[mn - 1]] == model.input[ij, ind_sortation_list[mn]])
# model.c.add(model.output[0,0] == model.input[0,1])
# model.c.add(model.output[1,0] == model.input[1,1])

# # #
# for ij in range(1, n):
#     model.c.add(sum([model.output[kl, ij] for kl in model.idx]) == sum([model.input[kl, ij] for kl in model.idx]))
# #
for ij in model.idx:    # check
    model.c.add(sum(model.charge[ij, kl] for kl in model.idx) == demand[ij])
#
# model.c.add(model.charge[0,0] + model.charge[0,1] == demand[0])
# model.c.add(model.charge[1,0] + model.charge[1,1] == demand[1])


for ij in model.idx:
    model.c.add(model.Y[ij] * energy >= sum(model.charge[kl, ij] for kl in model.idx))  # check
    model.c.add(model.Y[ij] <= model.X[ij] * 1000)  # check


# model.c.add(model.Y[0] * energy >= model.charge[0,0] + model.charge[1,0])
# model.c.add(model.Y[1] * energy >= model.charge[0,1] + model.charge[1,1])
# model.c.add(model.Y[0] <= model.X[0] * 1000)
# model.c.add(model.Y[1] <= model.X[1] * 1000)
for ij in model.idx:
    for kl in model.idx:
        model.c.add(model.input[ij, kl] >= 0)
        model.c.add(model.output[ij, kl] >= 0)
        model.c.add(model.charge[ij, kl] >= 0)
# check
# constraint to enforce imidiate coverage as soon as possible
#
# model.c.add(model.X[ij] * sum(model.charge[kl, ij] for kl in model.idx) >= model.X[ij] * (sum([energy_demand_matrix[kl, mn]
#                                                               for mn in model.idx for kl in model.idx if kl <= ij and mn <= ij] )))

model.obj = Objective(expr=((1/RBF)*(sum(model.X[ij] for ij in model.idx) * 50000 + sum(model.Y[ij] for ij in model.idx)
                                     * 10000)), sense=minimize)

opt = SolverFactory("gurobi")
opt_success = opt.solve(model)
time_of_optimization = time.strftime("%Y%m%d-%H%M%S")

X = np.array([model.X[ij].value for ij in model.idx])
Y = np.array([model.Y[ij].value for ij in model.idx])
charge = []
input = []
output = []

for ij in range(0, n):
        charge.append([model.charge[ij, kl].value for kl in model.idx])
        input.append([model.input[ij, kl].value for kl in model.idx])
        output.append([model.output[ij, kl].value for kl in model.idx])

charge = np.array(charge)
input = np.array(input)
output = np.array(output)

