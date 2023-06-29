#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Load PWD/output.txt, take line number LINE, calculate the expected cost using SHOTS samples and noise probability PE, put the result in PWD/noise_output.txt. Currently only works if the parameter multiplicity is 1, i.e. for One Parameter per Gate.
"""
from time import time
import sys

sys.path.insert(1, "../")
from datetime import datetime
import os
import argparse
import pickle
import numpy as np
import chainer as ch
import qem
import _HVQE
from scipy import stats
from matplotlib import pyplot as plt


class Name:
    pass


def get_command_line_input():
    parser = argparse.ArgumentParser(description="Compute noisy cost function values.")

    # Obligatory arguments
    parser.add_argument("path", type=str, help="path to data")
    parser.add_argument(
        "noise_type",
        type=str,
        help="Noise type: depol or bitflip. In earlier runs no noise type was specified. Then it is depol.",
    )
    parser.add_argument("pe", type=float, help="Probability of depolarizing error")
    parser.add_argument(
        "line",
        type=int,
        help="Load noiseless data from non-trivial line in output.txt with number LINE",
    )
    parser.add_argument(
        "ent_shots",
        type=int,
        help="Expected number of shots where an error occurs. (Expected non-trivial shots)",
    )

    # Parse the args.
    cmd_args = parser.parse_args()

    return cmd_args


if __name__ == "__main__":
    start = time()
    cmd_args = get_command_line_input()
    out = Name()
    data = _HVQE.load_data(cmd_args.path + "/output.txt")
    data_line = _HVQE.optimal_line(data, 140, 0, 1)

    # Prepare the init_reg, whose state is a dimer covering of run_args.complete_graph, as specified by run_args.init_layer.
    init_reg = qem.Reg(data_line[1]["n"])
    for edge in data_line[1]["init_layer"]:
        qem.apply_prepare_singlet(edge, init_reg)

    # Load the true ground state into memory for computation of infidelities.
    gs_reg = qem.Reg(data_line[1]["n"])
    with open(cmd_args.path + "/gs.dat", "rb") as file:
        gs_reg.psi.re = np.array(pickle.load(file)).reshape((2,) * data_line[1]["n"])

    print("initial infidelity", data_line[1]["inf_VQE"])

    # Compute the number of shots so that there will be an expected ent_shots non-trivial shots
    layers = data_line[1]["layers"]
    edges = [edge for layer in layers for edge in layer]
    d = data_line[0]["n_par"] / len(edges) * len(layers)
    assert d % 1 == 0
    d = int(d)
    out.shots = int(
        cmd_args.ent_shots / (1 - (1 - cmd_args.pe) ** (data_line[1]["n"] * (d + 1)))
    )
    print("Performing {} shots".format(out.shots))

    # Compute noisy energy and infidelity for loaded data point
    print("Computing sample no. ")
    with ch.no_backprop_mode():
        pars = ch.Variable(np.array(data_line[2]["opt_parameters"]))
        nntr_E_lst = (
            []
        )  # Noisy non-trivial list of energies. Contains the shots where a noise event occured only.
        nntr_inf_lst = []
        for s in range(out.shots):
            fn_args = [
                data_line[1]["complete_graph"],
                init_reg,
                data_line[1]["layers"],
                data_line[1]["n"],
                data_line[0]["par_multiplicity"],
                ch.Variable(np.array(data_line[2]["opt_parameters"])),
                cmd_args.pe,
                cmd_args.noise_type,
                gs_reg,
            ]
            (
                noisy_E,
                noisy_inf,
            ) = _HVQE.noisy_Heisenberg_energy_and_infidelity_from_parameters(*fn_args)

            if type(noisy_E) == ch.Variable:
                noisy_E = float(noisy_E.data)
            if type(noisy_inf) == ch.Variable:
                noisy_inf = float(noisy_inf.data)

            if noisy_E != "no_noise":
                nntr_E_lst.append(noisy_E)
                nntr_inf_lst.append(noisy_inf)

            print(s, end="\r", flush=True)

    # Do statistics on data
    # assert type(nntr_E_lst)==list
    # assert type(nntr_inf_lst)==list
    tr_shots = out.shots - len(nntr_E_lst)
    comp_E_lst = nntr_E_lst + [data_line[1]["E_VQE"]] * tr_shots
    # comp_inf_lst=nntr_inf_lst+[data_line[1]['inf_VQE']]*tr_shots
    #
    # try:
    #    stats.bootstrap
    #    stats.bayes_mvs
    # except AttributeError:
    #    print('In environment_GPU.yml, scipy is not updated to a version that includes bootstrap.. Try updating to en environment with scipy>=1.7.2 (conda update --all) or switching to environment_no_GPU.yml')

    ## Energy
    # out.noisy_E_mean=np.mean(comp_E_lst)
    # bsres=stats.bootstrap([comp_E_lst],np.mean, batch=1) # Bootstrap result
    # out.bootstrap_E_CI_low=bsres.confidence_interval.low # Confidence level of the mean of E.
    # out.bootstrap_E_CI_high=bsres.confidence_interval.high
    # out.bootstrap_E_st_error_CI=bsres.standard_error
    # ba_res=stats.bayes_mvs(comp_E_lst, alpha=0.95) # Bayesian result
    # out.bayes_E_CI_low=ba_res[0].minmax[0]
    # out.bayes_E_CI_high=ba_res[0].minmax[1]
    #
    ### Infidelity
    # out.noisy_inf_mean=np.mean(comp_inf_lst)
    # bsres=stats.bootstrap([comp_inf_lst],np.mean, batch=1) # Bootstrap result
    # out.bootstrap_inf_CI_low=bsres.confidence_interval.low # Confidence level of the mean of E.
    # out.bootstrap_inf_CI_high=bsres.confidence_interval.high
    # out.bootstrap_inf_st_error_CI=bsres.standard_error
    # ba_res=stats.bayes_mvs(comp_inf_lst,alpha=0.95) # Bayesian result
    # out.bayes_inf_CI_low=ba_res[0].minmax[0]
    # out.bayes_inf_CI_high=ba_res[0].minmax[1]

    # Export all the noisy non-trivial data (may be a very long array).
    # out.nntr_E_lst=nntr_E_lst
    # out.nnrr_inf_lst=nntr_inf_lst

    end = time()
    out.noisy_wall_clock = (end - start) / 60 / 60
    # out.noisy_date_end=str(datetime.utcnow())
    # data_line.append(vars(cmd_args))
    # data_line.append(vars(out))

    print("Finished noisy computation")

    # output=str(data_line)

    # Write input and results to disk. If no former output exists, print a line explaining the data in the output file.
    # if cmd_args.noise_type=='depol':
    #    if not os.path.exists('noisy_output.txt'):
    #        f=open('noisy_output.txt', 'w')
    #        f.write("### Noiseless and noisy output of the VQE is written to this file.\n")
    #    with open('noisy_output.txt', 'a') as f:
    #        f.write(output+'\n\n')
    # elif cmd_args.noise_type=='bitflip':
    #    if not os.path.exists('noisy_bitflip_output.txt'):
    #        f=open('noisy_bitflip_output.txt', 'w')
    #        f.write("### Noiseless and noisy bitflip output of the VQE is written to this file.\n")
    #    with open('noisy_bitflip_output.txt', 'a') as f:
    #        f.write(output+'\n\n')
    #
    # print(output)
    #

    #### Analyse distribution
    plt.hist(comp_E_lst)
    plt.savefig("thermal_state.pdf", bbox_inches="tight", pad_inches=0.005)
    plt.show()
