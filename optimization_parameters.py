"""

Definition of parameters for charging station allocation

"""
import pandas as pd

dir_0 = pd.read_csv(
    "data/resting_areas_dir_0.csv"
)  # file containing service stations for both directions + for "normal" direction
dir_1 = pd.read_csv(
    "data/resting_areas_dir_1.csv"
)  # file containing service stations for both directions + for "inverse" direction
n0 = len(dir_0)
n1 = len(dir_1)

g = 10000  # Maximum number of charging poles at one charging station
acc = 20  # (kWh) charged energy by a car
ec = 0.25  # (€/kWh) charging price for EV driver
e_tax = 0.15  # (€/kWh) total taxes and other charges
cfix = 50000  # (€) total installation costs of charging station installation
cvar1 = 10000  # (€) total installation costs of charging pole installation
eta = 0.003  # share of electric vehicles of car fleet
cars = 20  # number of charged cars per day at one charging pole
energy = acc * cars  # (kWh) charging energy per day by one charging pole
e_average = (
    (dir_0["Energiebedarf"].sum() + dir_1["Energiebedarf"].sum()) * eta / (n0 + n1)
)  # (kWh) average energy demand at a charging station
i = 0.05  # interest rate
T = 10  # (a) calculation period für RBF (=annuity value)
RBF = 1 / i - 1 / (i * (1 + i) ** T)  # (€/a) annuity value for period T


# extracting all highway names to create two additional columns: "first" and "last" to indicate whether resting areas
# are first resting areas along a singular highway in "normal" direction
dir_0["first"] = [False] * n0
dir_0["last"] = dir_0["first"]
dir_1["first"] = [False] * n1
dir_1["last"] = dir_1["first"]
l1 = dir_1.Autobahn.to_list()
l0 = dir_0.Autobahn.to_list()
l_ext = l0
l_ext.extend(l1)
highway_names = list(set(l_ext))
for name in highway_names:
    if name in l0:
        dir0_extract_indices = dir_0[dir_0.Autobahn == name].index
        dir_0.loc[dir0_extract_indices[0], "first"] = True
        dir_0.loc[dir0_extract_indices[-1], "last"] = True
    if name in l1:
        dir1_extract_indices = dir_1[dir_1.Autobahn == name].index
        dir_1.loc[dir1_extract_indices[0], "first"] = True
        dir_1.loc[dir1_extract_indices[-1], "last"] = True
