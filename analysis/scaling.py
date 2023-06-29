#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
import os
import sys
from scipy import optimize as op

sys.path.insert(1, "../")
import _HVQE

rc(
    "font", **{"family": "serif", "serif": ["Computer Modern"], "size": 9}
)  # Latex plot labels.
rc("text", usetex=True)

# For every threshold infidelity,for every n, for increasing p, take the point with lowest ENERGY, and look at the infidelity of that point.
# If this infidelity is below the threshold, add [n,p,calls] to the list.


def total_calls(data, n_par):  # Count the total number of function calls at n_par.
    calls = 0
    occ = 0
    for line in data:
        if line[0]["n_par"] == n_par:
            calls += line[2]["n_fn_calls"]
            occ += 1

    if occ == 32:
        return calls / 32  ## Retunrs calls per data point
    else:
        return None


lst = []
for th in [10**-m for m in range(1, 8, 1)]:
    sub_lst = [[2, 0, 0]]
    for n in range(4, 24, 2):
        data = _HVQE.load_data(
            "../data/chain/periodic/own_graph/opg/energy/{}/output.txt".format(n)
        )
        for n_par in range(n, 20 * n, n):
            line = _HVQE.optimal_line(data, n_par, 0, 1)
            if line != None:
                inf = line[1]["inf_VQE"]
                if inf <= th:
                    calls = line[2]["n_fn_calls"]
                    sub_lst.append([n, n_par / n, calls])
                    break
    sub_lst = np.array(sub_lst).transpose()
    lst.append(sub_lst)

## Plot the no. of function calls for all n as a fn of p.
lst3 = []
for n in range(4, 24, 2):
    lst2 = [[0, 0]]
    data = _HVQE.load_data(
        "../data/chain/periodic/own_graph/opg/energy/{}/output.txt".format(n)
    )
    for n_par in range(n, 12 * n, n):
        # line=_HVQE.optimal_line(data,n_par,0,1)
        # if line!=None:
        #    calls=line[2]['n_fn_calls']
        #    lst2.append([n_par/n,calls])
        calls = total_calls(data, n_par)
        lst2.append([n_par / n, calls])

    lst2 = np.array(lst2).transpose()
    lst3.append(lst2)

fig, ax = plt.subplots(2, 1, figsize=(4, 4))

for i in range(6, -1, -1):
    ax[0].plot(
        lst[i][0],
        lst[i][1],
        "-o",
        label="$10^{-{" + str(i + 1) + "}}$",
        alpha=0.7,
        markeredgecolor="none",
        markersize=5,
        solid_capstyle="round",
    )
ax[0].legend(title="$\mathcal I$", loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
ax[0].set_xticks(range(4, 24, 4))
ax[0].grid()
ax[0].set_xlabel("$n$")
ax[0].set_ylabel("$\overline{p}$")

####

for i in range(0, len(lst3)):
    ax[1].plot(
        lst3[i][0],
        lst3[i][1],
        "-o",
        label=str(i * 2 + 4),
        alpha=0.7,
        markeredgecolor="none",
        markersize=5,
        solid_capstyle="round",
    )


ax[1].set_xticks(range(0, 12, 2))
ax[1].set_yticks([0, 2 * 10**3, 4 * 10**3, 6 * 10**3], labels=[0, 2, 4, 6])
ax[1].text(0.02, 6700, "$*10^{3}$")
ax[1].grid()
ax[1].set_xlabel("$p$")
ax[1].set_ylabel("\# function calls")
# ax[1].set_ylim(top=3500)
#
#
####
xdata = lst3[8][0]
ydata = lst3[8][1]


def fun(x, a):
    return a * x**2


fit = op.curve_fit(fun, xdata, ydata, p0=1 / 19 * 1000)
yfit = fun(xdata, fit[0][0])
(bl_ln,) = ax[1].plot(xdata, yfit, "--", color="black", label="_")


###
l0 = ax[1].legend(title="$n$", loc="upper left", bbox_to_anchor=(1, 1.05), ncol=2)
l1 = ax[1].legend([bl_ln], ["$\sim x^2$"], loc="upper left", bbox_to_anchor=(1, 0.2))
ax[1].add_artist(l0)
ax[1].add_artist(l1)


fig.tight_layout()
prefix = "../../Postdoc/Papers/Variational quantum eigensolver for the Heisenberg antiferromagnet on the kagome lattice/v3"
fig.savefig(prefix + "/scaling.pdf", bbox_inches="tight", pad_inches=0.005)
