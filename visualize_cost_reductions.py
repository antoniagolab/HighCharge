import pandas as pd
import geopandas as gpd
import glob
import os
from shapely import wkt

# from optimization_parameters import *
from variable_definitions import *
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import pd2gpd
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

_filename = 'sensitivity_analyses\cost_decrease_potentials/cost_reduction_potentials.csv'
_cost_decrease_analysis = pd.read_csv(_filename)
scenario_file = pd.read_csv("scenarios/optimization_results_new_3.csv")

costs = [scenario_file.loc[3].costs] + _cost_decrease_analysis.costs.to_list()[0:4]
# spec_BEV_costs = [scenario_file.loc[3]['€/BEV']] + _cost_decrease_analysis['€/BEV'].to_list()[0:4]
# spec_kW_costs = [scenario_file.loc[3]['€/kW']] + _cost_decrease_analysis['€/kW'].to_list()[0:4]
labels = ['DT scenario'] + _cost_decrease_analysis['scenario_name'].to_list()[0:4]
labels = ['DT scenario\n 2030', 'Medium decrease\nin road traffic', 'Major decrease\nin road traffic', 'Increase\ndriving range', 'Increase\ncharging capacity']
c = '#f4f1de'

fig, ax = plt.subplots(figsize=(9, 4))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1], sharex=ax1)
ax.tick_params(axis='both', which='major', labelsize=10)
l2 = ax.bar(labels[0], costs[0], width=0.7, color=['#f6bd60'], zorder=10, label='infrastructure costs in DT scenario 2030')
l3 = ax.bar(labels[1:], costs[1:], width=0.7, color=['#3d405b'] * 4, zorder=10,
             label='reduced infrastructure costs of DT scenario')
ax.bar(labels, costs[0],  color=c, width=0.7, zorder=5)
ax.axhline(y=costs[0], linewidth=3, color='#f6bd60', linestyle='--', zorder=30)
ax.grid(axis="y")
ax.set_ylabel('Total infrastructure expansion costs (€)', fontsize=14, fontname="Franklin Gothic Book")
ax.text(labels[0], costs[0]/2,'€ ' + str(round((costs[0])/1e6, 1)) + ' Mio.', zorder=20, ha='center', va='center', fontsize=12, fontname="Franklin Gothic Book")
ax.set_yticklabels([str(e) + ' Mio.' for e in range(0, 70, 10)], fontsize=12, fontname="Franklin Gothic Book")
for ij in range(1, 5):
    ax.text(labels[ij], costs[ij] + (costs[0] - costs[ij])/2, u"\u2212" + ' € ' +
            str(round((costs[0] - costs[ij])/1e6, 1)) + ' Mio.', zorder=20, ha='center', va='center', fontsize=11, fontname="Franklin Gothic Book")
plt.subplots_adjust(hspace=.0)
ax.set_ylim([0, 70e6])
# ax2.grid()
# l0 = ax2.plot(labels, spec_kW_costs, marker='o', linestyle='dotted', color='#004733', linewidth=2, label="€/kW")
# ax3 = ax2.twinx()
# l1 = ax3.plot(labels, spec_BEV_costs, marker='o', linestyle='dotted', color='#0096c7', linewidth=2, label="€/BEV")
# ax2.set_ylim([120, 480])
# ax3.set_ylim([0, 100])

# insert text descriptions
# for ij in range(0, 5):
#     ax2.text(labels[ij], spec_kW_costs[ij] + 40, "{:.2f}".format(spec_kW_costs[ij]), va='top', color='#004733', ha='left')
#     ax3.text(labels[ij], spec_BEV_costs[ij] - 10, "{:.2f}".format(spec_BEV_costs[ij]), va='bottom', color='#0096c7',
#              ha='right')
# ax3.spines["left"].set_color("#004733")  # setting up Y-axis tick color to red
# ax3.spines["right"].set_color("#0096c7")  # setting up Y-axis tick color to red
# ax2.tick_params(axis="y", colors="#004733")
# ax3.tick_params(axis="y", colors="#0096c7")
#
# ax2.set_ylabel("€/kW",  color="#004733", fontsize=14)
# ax3.set_ylabel("€/BEV", rotation=-90, color="#0096c7", fontsize=14)

# adding labels to graph
y_size = 490
b_box = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5")
#
# for ij in range(0, len(labels)):
#     ax2.text(labels[ij], y_size, labels[ij], ha='center', va='top', bbox=b_box, fontweight='extra bold')
ax.xaxis.set_ticks_position('top')
# lns = l0 + l1
lns2 = [l2] + [l3]
labs = [l.get_label() for l in lns2]
ax.legend(lns2, labs, bbox_to_anchor=(0.5, -0.05))
# ax2.get_xaxis().set_ticks([])
# ax2.set_xticklabels(['' for e in range(0, len(labels))])
# ax1.xaxis.set_ticks_position('top')
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.tick_top()
# ax1.xaxis.set_ticks_position('both')
# plt.setp(ax3.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklabels(), visible=False)
ax.set_title('Cost-reduction potentials in the DT scenario 2030\n', fontsize=15, fontname="Franklin Gothic Book")
# plt.show()
plt.savefig('sensitivity_analyses/cost_decrease_potentials/cost_red.pdf', bbox_inches="tight")





