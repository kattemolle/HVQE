#!/usr/bin/env python3

"""
Make plots of noisy data.
"""
import sys
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
sys.path.insert(1, '../')
import _HVQE

rc('text', usetex=True)
rc('font',**{'family':'sans=serif','sans-serif':['ComputerModern']})


colors=[]
_,ax=plt.subplots()
for i in range(10):
    colors.append(next((ax._get_lines.prop_cycler))['color'])
del _,ax

####################

def check_noisy_data(data,noiseless_data,gates_per_cycle,p_max,occurences): # Check if all noisy points were obtained.
    pe2list=[line for line in data if line[3]['pe']==0.01]
    pe3list=[line for line in data if line[3]['pe']==0.001]
    pe4list=[line for line in data if line[3]['pe']==0.0001]
    pe5list=[line for line in data if line[3]['pe']==0.00001]

    for idx,lst in enumerate([pe2list,pe3list,pe4list,pe5list]):
        for n_par in range(gates_per_cycle,(p_max+1)*gates_per_cycle,gates_per_cycle):
            lst1=[line for line in lst if line[0]['n_par']==n_par]
            #assert len(lst1)==occurences

    # See which points are missing
    #for idx,line in enumerate(noiseless_data):
    #   E_VQE=line[1]['E_VQE']
    #   noisy_lines=[line for line in data if line[1]['E_VQE']==E_VQE]
    #   if len(noisy_lines)!=4:
    #       print(idx)

def en_plotter(data,E,load_pe,run_pe,gates_per_cycle,p_max,ax,error_bars=False,noiseless=False):
    plt_data=[]
    for n_par in range(0,(p_max+1)*gates_per_cycle,gates_per_cycle):
        try:
            opt_line=_HVQE.optimal_line(data,n_par,0,1,load_pe,run_pe)
            if run_pe==0:
                E_VQE=opt_line[1]['E_VQE']
                E_low=E_VQE
                E_high=E_VQE
            else:
                E_VQE=opt_line[4]['noisy_E_mean']
                E_low=opt_line[4]['bootstrap_E_CI_low']
                E_high=opt_line[4]['bootstrap_E_CI_high']

            rel_err=(E[0]-E_VQE)/E[0]
            rel_err_low=(E[0]-E_low)/E[0]
            rel_err_high=(E[0]-E_high)/E[0]
            p=n_par/gates_per_cycle
            plt_data.append([p,rel_err,rel_err_low,rel_err_high])
        except IndexError:
            pass

    if len(plt_data)!=0:
        plt_data=np.array(plt_data).transpose()
        if error_bars==False:
            style='-o'
        else:
            style='o'
        color_dict={'0':colors[0],'0.01':colors[5],'0.001':colors[4],'0.0001':colors[3],'1e-05':colors[2]}
        ax[0].semilogy(plt_data[0],plt_data[1],style,label=run_pe,markersize=4,alpha=0.75,color=color_dict[str(run_pe)])
        if run_pe!=0 and error_bars==True:
            ax[0].fill_between(plt_data[0],plt_data[2],plt_data[3])

def inf_plotter(data,load_pe,run_pe,gates_per_cycle,p_max,ax,error_bars=False,skip=False):
    plt_data=[]
    for n_par in range(0,(p_max+1)*gates_per_cycle,gates_per_cycle):
        p=n_par/gates_per_cycle
        try:
            opt_line=_HVQE.optimal_line(data,n_par,0,1,load_pe,run_pe)
            if run_pe==0:
                inf_VQE=opt_line[1]['inf_VQE']
                inf_low=inf_VQE
                inf_high=inf_VQE
                inf_ana=None
            else:
                inf_VQE=opt_line[4]['noisy_inf_mean']
                inf_low=opt_line[4]['bootstrap_inf_CI_low']
                inf_high=opt_line[4]['bootstrap_inf_CI_high']

                opt_nls_line=_HVQE.optimal_line(data,n_par,0,1,0,0)
                len_lay=len((opt_nls_line[1]['layers']))
                n=opt_nls_line[1]['n']
                er_loc=n*(len_lay*p+1)
                nls_inf=opt_nls_line[1]['inf_VQE']
                inf_ana=1-(1-run_pe)**er_loc*(1-nls_inf)

            plt_data.append([p,inf_VQE,inf_low,inf_high,inf_ana])
        except IndexError:
            pass

    if len(plt_data)!=0:
        plt_data=np.array(plt_data).transpose()
        if error_bars==False:
            style='-o'
        else:
            style='o'
        label_dict={'0':'0','0.01':'$10^{-2}$','0.001':'$10^{-3}$','0.0001':'$10^{-4}$','1e-05':'$10^{-5}$'}
        color_dict={'0':colors[0],'0.01':colors[5],'0.001':colors[4],'0.0001':colors[3],'1e-05':colors[2]}
        ax[1].semilogy(plt_data[0],plt_data[1],style,markersize=4,label=label_dict[str(run_pe)],alpha=0.75,color=color_dict[str(run_pe)])
        bl_ln,=ax[1].semilogy(plt_data[0],plt_data[4],color='black',label='_')

        if run_pe!=0 and error_bars==True:
            ax[1].fill_between(plt_data[0],plt_data[2],plt_data[3])

        return bl_ln



def plotter(prefix,out_prefix,fname,gates_per_cycle,p_max,occurences,bitflip=False,noise_aware=False,error_bars=False):
    print('plotting',fname)
    if bitflip==False:
        data_path=prefix+'noisy_output.txt'
    elif bitflip==True:
        data_path=prefix+'noisy_bitflip_output.txt'
    data_path_noiseless=prefix+'output.txt'
    E_path=prefix+'lowest_energies.txt'
    data=_HVQE.load_data(data_path)
    noiseless_data=_HVQE.load_data(data_path_noiseless)
    E=_HVQE.get_E_lst(E_path)
    check_noisy_data(data,noiseless_data,gates_per_cycle,p_max,occurences)

    fig,ax=plt.subplots(1,2,figsize=(6,3))

    for pe in [0.01,0.001,0.0001,0.00001]:
        en_plotter(data,E,0,pe,gates_per_cycle,p_max,ax,error_bars)
        if noise_aware==True:
            en_plotter(data,E,pe,pe,gates_per_cycle,p_max,ax,error_bars)

    en_plotter(data,E,0,0,gates_per_cycle,p_max,ax,error_bars)

    for pe in [0.01,0.001,0.0001,0.00001]:
        bl_ln=inf_plotter(data,0,pe,gates_per_cycle,p_max,ax,error_bars)
        if noise_aware==True:
            bl_ln=inf_plotter(data,pe,pe,gates_per_cycle,p_max,ax,error_bars)

    inf_plotter(data,0,0,gates_per_cycle,p_max,ax)

    rel_err_exited=(E[0]-E[1])/E[0]
    ax[0].axhline(y=rel_err_exited,zorder=-100,color=colors[1])

    ax[0].set_ylabel('Relative energy error $\mathcal E$')
    ax[0].set_title('Energy using energy')
    ax[0].set_xlabel('$p$')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True)

    ax[1].set_ylabel('Infidelity $\mathcal I$')
    ax[1].set_title('Infidelity using energy')
    ax[1].set_xlabel('$p$')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid(True)
    l0=ax[1].legend(title='$p_e$ (sim.)',handlelength=1.5)
    l1 = ax[1].legend([bl_ln],['ana.'],loc='lower center',bbox_to_anchor=(0.36, 0., 0.36, 0.5))
    ax[1].add_artist(l0)
    ax[1].add_artist(l1)

    fig.tight_layout()

    #plt.show()

    fig.savefig(out_prefix+fname+'.pdf',bbox_inches='tight',pad_inches=0.005)

    return fig


if __name__=='__main__':
    out_prefix='../../Postdoc/Papers/Variational quantum eigensolver for the Heisenberg antiferromagnet on the kagome lattice/v3/'

    fname='noisy_KVQE_G'
    plotter('../data/kagome/open/on_grid/opg/energy/2x2/',out_prefix,fname,30,16,10)

    fname='noisy_KVQE_K'
    plotter('../data/kagome/periodic/own_graph/opg/energy/3x2/',out_prefix,fname,36,37,10)

    fname='noisy_CVQE'
    plotter('../data/chain/periodic/own_graph/opg/energy/20/',out_prefix,fname,20,11,32)

    # Some extra plots, not in paper
    #fname='noisy_bitflip_KVQE_G'
    #plotter('../data/kagome/open/on_grid/opg/energy/2x2/',out_prefix,fname,30,16,10,both=True)

    #fname='noisy_bitflip_KVQE_K'
    #plotter('../data/kagome/periodic/own_graph/opg/energy/3x2/',out_prefix,fname,36,37,10,both=True)

    #fname='noisy_bitflip_CVQE'
    #plotter('../data/chain/periodic/own_graph/opg/energy/20/',out_prefix,fname,20,11,32,both=True)
