"""

Created by Antonia Golab

Optimization of charging station allocation

last edit: 2021/12/08

"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from optimization_parameters import *


def optimization():
    """
    Constraint and objective function definition + solution of optimization using parameters defined in
    optimization_parameters.py

    :return:
    """
    model = ConcreteModel()  # Model initialization

    # --------------------------------------------- variable definition ---------------------------------------------
    # driving directions "normal" (=0) and "inverse" (=1) are treated separately throughout the optimization model
    # charging demand at each resting area is then summed up if it is for both directions

    model.IDX_0 = range(n0)
    model.IDX_1 = range(n1)

    # binary variables stating whether a charging station is installed at given ra (=resting area)
    model.pXi_dir_0 = Var(model.IDX_0, within=Binary)
    model.pXi_dir_1 = Var(model.IDX_1, within=Binary)

    # integer variables indicating number of charging poles at a charging station
    model.pYi_dir_0 = Var(
        model.IDX_0,
        within=Integers,
    )
    model.pYi_dir_1 = Var(model.IDX_1, within=Integers)

    # a fictitious energy load is passed on from ra to ra along a highway which (pE_Zu) which is the difference between
    # energy demand and covered energy demand at a resting area (pE_Laden); the remaining energy is passed on to the
    # next ra (pE_Ab)
    model.pE_Zu_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_Zu_1 = Var(model.IDX_1, within=NonNegativeReals)
    model.pE_Ab_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_Ab_1 = Var(model.IDX_1, within=NonNegativeReals)
    model.pE_Laden_0 = Var(model.IDX_0, within=NonNegativeReals)
    model.pE_Laden_1 = Var(model.IDX_1, within=NonNegativeReals)

    # ------------------------------------------------- constraints -------------------------------------------------

    # the pE_Zu needs to be 0 at each first ra as it is assumed that no energy demand is previously covered;
    # all energy demand is constrained to be covered at the last ra on a highway
    model.c1 = ConstraintList()

    firsts_0 = dir_0[dir_0["first"] == True].index
    for ind in firsts_0:
        model.c1.add(model.pE_Zu_0[ind] == 0)
    lasts_0 = dir_0[dir_0["last"] == True].index
    for ind in lasts_0:
        model.c1.add(model.pE_Ab_0[ind] == 0)

    firsts_1 = dir_1[dir_1["first"] == True].index
    for ind in firsts_1:
        model.c1.add(model.pE_Zu_1[ind] == 0)
    lasts_1 = dir_1[dir_1["last"] == True].index
    for ind in lasts_1:
        model.c1.add(model.pE_Ab_1[ind] == 0)

    # constraints defining the passing on of the fictitious energy
    model.loading = ConstraintList()
    for ij in range(1, n0):
        model.loading.add(model.pE_Zu_0[ij] == model.pE_Ab_0[ij - 1])

    for ij in range(1, n1):
        model.loading.add(model.pE_Zu_1[ij] == model.pE_Ab_1[ij - 1])

    # constraining the fictitiously passed on energy to ensure evenly distributed charging network
    model.c5 = ConstraintList()
    for ij in model.IDX_0:
        model.c5.add(model.pE_Zu_0[ij] <= e_average)

    model.c6 = ConstraintList()
    for ij in model.IDX_1:
        model.c6.add(model.pE_Zu_1[ij] <= e_average)

    model.c7 = ConstraintList()
    for ij in model.IDX_0:
        model.c7.add(model.pE_Ab_0[ij] <= e_average)

    model.c8 = ConstraintList()
    for ij in model.IDX_1:
        model.c8.add(model.pE_Ab_1[ij] <= e_average)

    # defining the relationship between energy demand, demand coverage and net energy demand
    model.c9 = ConstraintList()
    for ij in model.IDX_0:
        model.c9.add(
            model.pE_Laden_0[ij]
            - dir_0.Energiebedarf.to_list()[ij] * eta
            - model.pE_Zu_0[ij]
            + model.pE_Ab_0[ij]
            == 0
        )

    model.c10 = ConstraintList()
    for ij in model.IDX_1:
        model.c10.add(
            model.pE_Laden_1[ij]
            - dir_1.Energiebedarf.to_list()[ij] * eta
            - model.pE_Zu_1[ij]
            + model.pE_Ab_1[ij]
            == 0
        )

    # defining how much energy demand a charging station is able to cover
    model.c11 = ConstraintList()
    for ij in model.IDX_0:
        model.c11.add(model.pYi_dir_0[ij] * energy >= model.pE_Laden_0[ij])

    model.c12 = ConstraintList()
    for ij in model.IDX_1:
        model.c12.add(model.pYi_dir_1[ij] * energy >= model.pE_Laden_1[ij])

    # installing a constraint preferring charging station at ras for both directions
    model.c21 = ConstraintList()
    count = len(dir_0[(dir_0.Richtung == 2) & (dir_0.Energiebedarf > energy / eta)])
    inds_pref = dir_0[
        (dir_0.Richtung == 2) & (dir_0.Energiebedarf > energy / eta)
    ].index
    for ind in inds_pref:
        name = dir_0.Name.to_list()[ind]
        dir_1_inds = dir_1[dir_1.Name == name].index
        model.c21.add(model.pXi_dir_0[ind] == 1)
        model.c21.add(model.pYi_dir_0[ind] >= 1)
        model.c21.add(model.pXi_dir_1[dir_1_inds[0]] == 1)
        model.c21.add(model.pYi_dir_1[dir_1_inds[0]] >= 1)

    # setting a maximum number at each station + defining relation between the binary variable defining whether a
    # resting area has a charging station and the number of charging poles at one
    model.c19 = ConstraintList()
    for ij in model.IDX_0:
        model.c19.add(model.pYi_dir_0[ij] <= g * model.pXi_dir_0[ij])

    model.c20 = ConstraintList()
    for ij in model.IDX_1:
        model.c19.add(model.pYi_dir_1[ij] <= g * model.pXi_dir_1[ij])

    # ------------------------------------------------- objective -------------------------------------------------
    # maximization of revenue during the observed observation period

    model.obj = Objective(
        expr=(
            sum(model.pE_Laden_0[n] * (ec - e_tax) for n in model.IDX_0)
            + sum(model.pE_Laden_1[n] * (ec - e_tax) for n in model.IDX_1)
            - (
                1
                / RBF
                * 1
                / 365
                * (
                    (
                        sum(model.pXi_dir_0[n] for n in model.IDX_0)
                        + sum(model.pYi_dir_1[n] for n in model.IDX_1)
                        - count
                    )
                    * cfix
                    + (
                        sum(model.pYi_dir_0[n] for n in model.IDX_0)
                        + sum(model.pYi_dir_1[n] for n in model.IDX_1)
                    )
                    * cvar1
                )
            )
        ),
        sense=maximize,
    )

    # ------------------------------------------------- solution --------------------------------------------------

    opt = SolverFactory("gurobi")
    opt_success = opt.solve(model)

    # -------------------------------------------------- results --------------------------------------------------

    pYi_dir_0 = np.array([model.pYi_dir_0[n].value for n in model.IDX_0])
    pYi_dir_1 = np.array([model.pYi_dir_1[n].value for n in model.IDX_1])
    pE_Laden_0 = np.array([model.pE_Laden_0[n].value for n in model.IDX_0])
    pE_Laden_1 = np.array([model.pE_Laden_1[n].value for n in model.IDX_1])

    umsatz = sum(pE_Laden_0 * ec) + sum(pE_Laden_1 * ec)

    print(f"Total revenue per year: € {umsatz*365}")
    print(f"Total amount of charging poles: {sum(pYi_dir_0)+sum(pYi_dir_1)}")


if __name__ == "__main__":
    optimization()
