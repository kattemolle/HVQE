#!/usr/bin/env python3

import numpy
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
import matplotlib as mpl
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rc("text", usetex=True)
rc("font", **{"family": "sans=serif", "sans-serif": ["ComputerModern"]})

n_exited = 1
colors = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]


def import_data(cost_fn, prefix):
    # Import exact energies
    with open(prefix + "lowest_energies.txt", "r") as f:
        E = f.readlines()
        E = [eval(x.strip()) for x in E]

    # Import data
    with open(prefix + "output.txt", "r") as f:
        f.readline()
        data = f.readlines()

    data = [line for line in data if line != "\n"]
    data = [eval(line.strip()) for line in data]
    data = [line for line in data if line[0]["n_iter"] == 0]
    par_multiplicity = data[0][0]["par_multiplicity"]
    assert all(
        [line[0]["par_multiplicity"] == par_multiplicity for line in data]
    ), "different par-multiplicities in one file."

    gates_per_cycle = len(complete_graph)

    data.sort(key=lambda x: x[0]["n_par"])
    p_list = [line[0]["n_par"] * par_multiplicity / gates_per_cycle for line in data]
    E_VQE_list = [-(line[1]["E_VQE"] - E[0]) / E[0] for line in data]
    inf_VQE_list = [line[1]["inf_VQE"] for line in data]
    assert len(p_list) == len(E_VQE_list) == len(inf_VQE_list)
    data = [[p_list[i], E_VQE_list[i], inf_VQE_list[i]] for i in range(len(p_list))]

    return p_list, E_VQE_list, inf_VQE_list, data, E


def plot_VQE_data(ax, cost_fn, plot_fn, prefix):
    p_list, E_VQE_list, inf_VQE_list, data, E = import_data(cost_fn, prefix)

    # Generate lists for drawing the lines
    p_set = set(p_list)
    p_set = list(p_set)
    min_E_list = []
    min_inf_list = []
    if cost_fn == "energy":
        for i in p_set:
            E_group = [E_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
            inf_group = [inf_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
            min_E = min(E_group)
            min_index = E_group.index(min_E)
            min_E_list.append(min_E)
            min_inf_list.append(inf_group[min_index])

    if cost_fn == "infidelity":
        for i in p_set:
            E_group = [E_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
            inf_group = [inf_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
            min_inf = min(inf_group)
            min_index = inf_group.index(min_inf)
            min_inf_list.append(min_inf)
            min_E_list.append(E_group[min_index])

    # sort the lists
    together = [[p_set[i], min_E_list[i], min_inf_list[i]] for i in range(len(p_set))]
    together = sorted(together, key=lambda x: x[0])
    p_set = [together[i][0] for i in range(len(p_set))]
    min_E_list = [together[i][1] for i in range(len(p_set))]
    min_inf_list = [together[i][2] for i in range(len(p_set))]

    if plot_fn == "energy":
        ax.semilogy(p_set, min_E_list, zorder=1000, color=colors[0])
        ax.scatter(p_list, E_VQE_list, alpha=0.1, edgecolors="none")
        for i in range(1, n_exited + 1):
            ax.axhline(
                y=-(E[i] - E[0]) / E[0], alpha=1, color=colors[1], zorder=-100
            )  # Plot a horizontal line at the ith excited state.
        ax.set_ylabel("Relative energy error $\mathcal E$")

        ax.set_title("Energy using " + cost_fn)

    if plot_fn == "infidelity":
        ax.semilogy(p_set, min_inf_list, zorder=1000, color=colors[0])
        ax.scatter(p_list, inf_VQE_list, alpha=0.1, edgecolors="none")
        ax.set_ylabel("Infidelity $\mathcal I$")
        ax.set_title("Infidelity using " + cost_fn)

    elif plot_fn == "wall_clock":
        wall_clock_list = [line[1]["wall_clock"] for line in n_iter_class]
        ax.semilogy(p_list, wall_clock_list, "o")
        ax.set_ylabel("Wall-clock time (h)")

    # On the x-axis, put the number of cycles rather than the number of parameters.

    ax.set_xlabel("$p$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)


def plot_inset(in_what_ax):
    till = 4
    cost_fn = "energy"
    p_list, E_VQE_list, inf_VQE_list, data, E = import_data(cost_fn)
    ax = inset_axes(in_what_ax, width=0.75, height=0.75)

    # Generate lists for drawing the lines
    p_set = set(p_list)
    p_set = list(p_set)
    min_E_list = []
    min_inf_list = []
    inf_list = []

    for i in p_set:
        E_group = [E_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
        inf_group = [inf_VQE_list[j] for j in range(len(p_list)) if p_list[j] == i]
        inf_list.append(inf_group)
        min_E = min(E_group)
        min_index = E_group.index(min_E)
        min_E_list.append(min_E)
        min_inf_list.append(inf_group[min_index])

    # sort the lists
    together = [
        [p_set[i], min_E_list[i], min_inf_list[i], inf_list[i]]
        for i in range(len(p_set))
    ]
    together = sorted(together, key=lambda x: x[0])
    p_set = [together[i][0] for i in range(len(p_set))]
    min_E_list = [together[i][1] for i in range(len(p_set))]
    min_inf_list = [together[i][2] for i in range(len(p_set))]
    inf_list = [together[i][3] for i in range(len(p_set))]

    ax.semilogy(p_set[:till], min_inf_list[:till], zorder=1000, color=colors[0])
    for i in range(till):
        ax.scatter(
            [p_set[i]] * len(inf_list[i]),
            inf_list[i],
            alpha=0.1,
            edgecolors="none",
            c=colors[0],
        )
    # On the x-axis, put the number of cycles rather than the number of parameters.
    ax.set_xlabel("$p$")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0.6, 0.8, 0.9999], minor=True)
    ax.set_yticklabels(["0.6", "0.8", "1.0"], minor=True)


##################################################

prefix = "../data/kagome/periodic/own_graph/opg/energy/3x2/"
with open(prefix + "graph_input.txt", "r") as f:
    exec(f.read())
complete_graph = complete_graph_input
del complete_graph_input
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_VQE_data(axs[0], "energy", "energy", prefix)
plot_VQE_data(axs[1], "energy", "infidelity", prefix)
fig.tight_layout()
fig.savefig("kagome_on_kagome_plot.pdf", bbox_inches="tight", pad_inches=0)

prefix = "../data/chain/periodic/own_graph/opg/energy/20/"
with open(prefix + "graph_input.txt", "r") as f:
    exec(f.read())
complete_graph = complete_graph_input
del complete_graph_input
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_VQE_data(axs[0], "energy", "energy", prefix)
plot_VQE_data(axs[1], "energy", "infidelity", prefix)
fig.tight_layout()
fig.savefig(
    "../../Postdoc/Papers/Variational quantum eigensolver for the Heisenberg antiferromagnet on the kagome lattice/v3/chain_plot.pdf",
    bbox_inches="tight",
    pad_inches=0.005,
)

prefix = "../data/kagome/open/on_grid/opg/energy/2x2/"
with open(prefix + "graph_input.txt", "r") as f:
    exec(f.read())
complete_graph = complete_graph_input
del complete_graph_input
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_VQE_data(axs[0], "energy", "energy", prefix)
plot_VQE_data(axs[1], "energy", "infidelity", prefix)
fig.tight_layout()
fig.savefig(
    "../../Postdoc/Papers/Variational quantum eigensolver for the Heisenberg antiferromagnet on the kagome lattice/v3/kagome_on_grid_plot.pdf",
    bbox_inches="tight",
    pad_inches=0.005,
)
