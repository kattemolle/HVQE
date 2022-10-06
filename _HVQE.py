"""
Contains function defs for running HVQE.py
"""
class Name: # Simple namespace class that is used for dumping and restarting the program.
    pass

import numpy # For the cases where GPU==True and we still want to use numpy.
import qem
import chainer as ch
from datetime import datetime
import argparse
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

try: # Use GPU if CuPy installation is available. 
    import cupy as xp
except ImportError:
    import numpy as xp

def get_command_line_input():
    parser=argparse.ArgumentParser(description='Run a VQE using the quantum emulator qem.')
    
    # Obligatory arguments
    parser.add_argument('path',type=str,help='The path to the folder in which the file graph_input.txt is located. Output files will be placed in the same folder.')
    parser.add_argument('n_par',type=int,help='The number of parameters to be used in the VQE.')
    parser.add_argument('par_multiplicity',type=int,help="The parameter multiplicity; the number of gates that share the same parameter. Parameters are given to the gates in the order in which edges apear in the (flattend version of) the list 'layers_input', which should be defined in the file path/graph_input.txt")

    # Optional arguments
    parser.add_argument('--n_iter','-i',type=int,default=0,help='Number of iterations of the basinhopping routine.')
    parser.add_argument('--cost_fn','-c',type=str,default='energy',help="The cost function that the VQE tries to minimize. Should be 'energy' or 'infidelity'. Note that on a quantum computer it would be impractical to use the infidelity as a cost function.")
    parser.add_argument('--temperature','-T',type=float,default=1.,help="Temperature for the metropolis creterion in the scipy basinhopping routine.")
    parser.add_argument('--stepsize','-s',type=float,default=1.,help="The max stepsize of random displacement per parameter after each local optimization in the scipy basinhopping routine.")
    parser.add_argument('--init_par',type=float,nargs='+',default=None,help='A list of initial parameters from which the basinhopping routine starts. Use as: --init_par par1 par2 par3 ... . If no --init_par is given, random parameters will be chosen in a specific interval. These are exported to output.txt at the end of the script for reproducibility. See VQE.run_VQE for more info.')
    parser.add_argument('--dump_interval', type=int,default=None, help='Dump the state of the program to path/dump.dat every dump_interval function calls.')

    # Parse the args.
    cmd_args=parser.parse_args()
    if cmd_args.path[-1]=='/':
        cmd_args.path=cmd_args.path[:-1]

    if cmd_args.par_multiplicity==0:
        cmd_args.par_multiplicity=1

    return cmd_args

def load_data(path):
    """
    Load data from path (a output.txt or noisy_output.txt).
    """
    with open(path,'r') as f:
        f.readline()
        _data=f.readlines()
    data=[]
    for line_nr,line in enumerate(_data):
        if line !='\n':
            try:
                new_line=eval(line.strip())
                data.append(new_line)
            except:
                print('WARNING: corruption found in line {} of {}'.format(line_nr+2,path))

    return data

def get_E_lst(path):
    """
    Load lowest energies from path (a lowest_energy.txt)
    """
    with open(path,'r') as f:
        Elst=f.readlines()
        Elst=[eval(x.strip()) for x in Elst]
        Elst=[float(E) for E in Elst]
        Elst.sort()

    return Elst

def optimal_line(data,n_par,n_iter,par_multiplicity,load_pe=0,run_pe=0):
    """
    Find optimal line in data with given n_par, n_iter, pe.
    """
    _data=[line for line in data if line[0]['n_par']==n_par and line[0]['n_iter']==n_iter and line[0]['par_multiplicity']==par_multiplicity]
    if load_pe==0 and run_pe==0:
        _data.sort(key=lambda x: x[1]['E_VQE'])
    elif load_pe==0 and run_pe!=0:
        _data=[line for line in _data if line[3]['pe']==run_pe]
        _data.sort(key=lambda x: x[1]['E_VQE'])
    elif load_pe!=0 and run_pe!=0:
        assert load_pe==run_pe
        _data=[line for line in _data if line[3]['pe']==run_pe]
        _data.sort(key=lambda x: x[4]['noisy_E_mean'])

    if len(_data)==0:
        print('optimal_line: no entries found for n_par={}, n_iter={}, load_pe={}, run_pe={} in data'.format(n_par,n_iter,load_pe,run_pe))
        return None
    else:
        return _data[0]

def get_n_par_list(path,n_iter,par_multiplicity,pe):
    """
    Get the list of available n_par values in path (a noise_output.txt) with given n_iter,par_multiplicity, and pe.
    """
    assert pe!=0, 'Only implemented for noisy output.'
    def filter_data(data,n_iter,par_multiplicity,pe):
        def conditions(line):
            cond=[line[0]['n_iter']==n_iter,
                  line[0]['par_multiplicity']==par_multiplicity,
                  line[3]['pe']==pe
                  ]
            return all(cond)
        filtered=[line for line in data if conditions(line)]
        return filtered

    data=load_data(path)
    data=filter_data(data,n_iter,par_multiplicity,pe)
    n_par_data=list(set([line[0]['n_par'] for line in data]))
    n_par_data.sort()
    return n_par_data

def Heisenberg_energy_from_parameters(complete_graph,init_reg,layers,n,par_multiplicity,parameters):
    """
    Return the energy of a state as defined via the init state, the ansatz and a setting for the parameters. Ansatz must already be mapped to ints and given as regular python list.
    
    Returns
    -------    E : chainer.Variable
    
    """
    reg=qem.EmptyReg(n)
    reg.psi=init_reg.psi

    edges=[edge for layer in layers for edge in layer]

    for i in range(len(parameters)):
        gate=qem.Heisenberg_exp(parameters[i])
        for j in range(par_multiplicity):
            edge=edges[(i*par_multiplicity+j)%len(edges)]
            action=qem.Action(edge,gate)
            qem.apply_action(action,reg)

    E=qem.Heisenberg_energy(complete_graph,reg)
    return E

def noisy_Heisenberg_energy_and_infidelity_from_parameters(complete_graph,init_reg,layers,n,par_multiplicity,parameters,pe,noise_type,gs_reg):
    """
    Same as Heisenberg_energy_from_parameters, but at every qubit and, before the first layer, and after layer, with probability pe, a Pauli X Y or Z will be inserted uniformly at random.
    Thus, the output becomes stochasic.

    Returns
    -------
    [E,inf] : [chainer.Variable, chainer.Variable]

    """
    assert par_multiplicity==1, 'Error functionality is only available for One parameter Per Gate (OPG), i.e. par_multiplicity=1'
    reg=qem.EmptyReg(n)
    reg.psi=init_reg.psi

    edges=[edge for layer in layers for edge in layer]
    assert len(parameters)%len(edges)==0

    tot_layers=len(parameters)/len(edges)*len(layers)
    tot_error_loc=n*(1+tot_layers)
    assert tot_error_loc%1==0
    tot_error_loc=int(tot_error_loc)
    error_list=numpy.random.rand(tot_error_loc)
    if all(numpy.greater_equal(error_list,pe)):
        return 'no_noise','no_noise'

    i=0
    j=0

    # Apply error to init state probabilisticly
    if noise_type=='depol':
        for q in range(n):
            r=error_list[j]
            if r<pe/3:
                qem.apply_X((q,),reg)
            elif r<2*pe/3:
                qem.apply_Y((q,),reg)
            elif r<pe:
                qem.apply_Z((q,),reg)
            j+=1
    elif noise_type=='bitflip':
        for q in range(n):
            r=error_list[j]
            if r<pe:
                qem.apply_X((q,),reg)
            j+=1

    while i<len(parameters):
        for layer in layers:
            for edge in layer:
                gate=qem.Heisenberg_exp(parameters[i])
                i+=1
                action=qem.Action(edge,gate)
                qem.apply_action(action,reg)
            # Apply error after circuit layer probabilisticly
            if noise_type=='depol':
                for q in range(n):
                    r=error_list[j]
                    if r<pe/3:
                        qem.apply_X((q,),reg)
                    elif r<2*pe/3:
                        qem.apply_Y((q,),reg)
                    elif r<pe:
                        qem.apply_Z((q,),reg)
                    j+=1
            elif noise_type=='bitflip':
                for q in range(n):
                    r=error_list[j]
                    if r<pe:
                        qem.apply_X((q,),reg)
                    j+=1

    assert j==tot_error_loc, 'Not all sampled errors were applied'
    E=qem.Heisenberg_energy(complete_graph,reg)
    inf=qem.infidelity(reg,gs_reg)
    return E,inf

def infidelity_from_parameters(init_reg,layers,n,par_multiplicity,parameters,gs_reg):
    reg=qem.EmptyReg(n)
    reg.psi=init_reg.psi    

    edges=[edge for layer in layers for edge in layer]

    for i in range(len(parameters)):
        gate=qem.Heisenberg_exp(parameters[i])
        for j in range(par_multiplicity):
            edge=edges[(i*par_multiplicity+j)%len(edges)] 
            action=qem.Action(edge,gate)
            qem.apply_action(action,reg)

    inf=qem.infidelity(reg,gs_reg)
    return inf

def run_VQE(cmd_args,run_args,init_reg,gs_reg):
    """
    Run the VQE.
    """
    vqe_out=Name()
    vqe_out.n_fn_calls=0
    vqe_out.local_min_list=[]
    vqe_out.local_min_parameters_list=[]
    vqe_out.local_min_accept_list=[]
    
    def calc_cost(parameters):
        nonlocal vqe_out
        nonlocal cmd_args
        nonlocal run_args
        tmp=Name()
        parameters=ch.Variable(xp.array(parameters))
        
        if cmd_args.cost_fn=='energy':
            cost=Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,parameters)
        elif cmd_args.cost_fn=='infidelity':
            cost=infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,parameters,gs_reg)
        else:
            raise ValueError('Not a valid cost function')
        cost.backward()
        g=parameters.grad
        vqe_out.n_fn_calls+=1
        print('.',end='',flush=True) #Progress indicator. One dot per function call. Here one function call is defined as one forward and one backward evaluation. 
        if run_args.GPU==True:
            cost=cost.array.get()
            g=g.get()
        elif run_args.GPU==False:
            cost=cost.array

        ### Dump state of the program. Restart has to be done by hand by running another HVQE.py from the command line. 
        if cmd_args.dump_interval!=None:
            if vqe_out.n_fn_calls%cmd_args.dump_interval==0:
                tmp=Name()
                tmp.parameters=parameters.array.tolist()
                tmp.cost=float(cost)
                tmp.g=g.tolist()
                date_dump=str(datetime.utcnow()) # Current time in UTC.
                vqe_out.init_par=list(vqe_out.init_par)
                dump=[vars(cmd_args),vars(run_args),vars(vqe_out),vars(tmp)]
                with open(cmd_args.path+'/dump.txt', 'a') as file:
                    file.write(str(dump)+'\n\n')
                print('Data dump on', date_dump)
        ###
        return cost, g

    def callback(x,f,accept): #If basinhopping is run with n_iter=0, only a single local minimum is found, and in this case the value of the cost function, and the parameters, are in fact stored, because this minimum is the optimal minimum found and delivers the data for the output of the bassinhopping routine as a whole.
        nonlocal vqe_out
        print('\nNew local min for', vars(cmd_args))
        print('cost=',float(f),'accepted=',accept,'parameters=',list(x))
        vqe_out.local_min_list.append(float(f))
        vqe_out.local_min_parameters_list.append(list(x))
        vqe_out.local_min_accept_list.append(accept)
       
    if cmd_args.init_par is None:
        vqe_out.init_par=numpy.random.rand(cmd_args.n_par)/1000-1/2000
    else:
        assert len(cmd_args.init_par)==cmd_args.n_par, 'List of initial parameters must be of length n_par.'
        vqe_out.init_par=numpy.array(cmd_args.init_par)
    if cmd_args.n_par==0: # If there is no circuit, just output the energy of the init state.
        if cmd_args.cost_fn=='energy':
            vqe_out.cost_VQE=Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,[]) # Still a Chainer.Variable
            vqe_out.cost_VQE=float(vqe_out.cost_VQE.array)
            vqe_out.opt_parameters=[]
            vqe_out.init_par=[]
        if cmd_args.cost_fn=='infidelity':
            vqe_out.cost_VQE=infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,[],gs_reg)
            vqe_out.cost_VQE=float(vqe_out.cost_VQE.array)
            vqe_out.opt_parameters=[]
            vqe_out.init_par=[]
            
    else:
        sol=scipy.optimize.basinhopping(calc_cost,vqe_out.init_par,stepsize=cmd_args.stepsize,minimizer_kwargs={'jac':True},niter=cmd_args.n_iter,interval=25,callback=callback,T=cmd_args.temperature)
        vqe_out.cost_VQE=float(sol.fun)
        vqe_out.opt_parameters=sol.x.tolist()
        vqe_out.init_par=list(vqe_out.init_par)

    return vqe_out
     
def plot_VQE_data(path,fn,par_multiplicity,gates_per_cycle):
    # Import data
    with open(path+'/output.txt','r') as f:
        f.readline()
        data=f.readlines()
        data=[line for line in data if line != '\n']
        data=[eval(x.strip()) for x in data]
        
    E=0
    if fn=='energy':    
        with open(path+'/lowest_energies.txt','r') as f:
            E=f.readlines()
            E=[eval(x.strip()) for x in E]
            E[0]=E[0]

    # Sort data into sublists based on the value of n_iter
    n_iter_set=set([line[0]['n_iter'] for line in data])
    data_=[]
    for n_iter in n_iter_set:
        n_iter_array=[line for line in data if line[0]['n_iter']==n_iter]
        n_iter_array.sort(key=lambda x: x[0]['n_par']) # Put datapoints in order of increasing number of parameters. 
        data_.append(n_iter_array)

    data=data_
    n_iter_set=list(n_iter_set)
    
    # Make one plot for every possible val of n_iter, all in one figure.
    fig, ax = plt.subplots()
    for n_iter_class in data:
        n_par_list=[line[0]['n_par'] for line in n_iter_class]
        p_list=[n_par*par_multiplicity/gates_per_cycle for n_par in n_par_list]
        if fn=='energy':
            E_VQE_list=[line[1]['E_VQE'] for line in n_iter_class]
            E_VQE_list=[-(E_VQE-E[0])/E[0] for E_VQE in E_VQE_list] # The relative error in the energy is going to be plotted.
            ax.semilogy(p_list,E_VQE_list,'-o')
        elif fn=='infidelity':
            inf_VQE_list=[line[1]['inf_VQE'] for line in n_iter_class]
            ax.semilogy(p_list,inf_VQE_list,'-o')
        elif fn=='wall_clock':
            wall_clock_list=[line[1]['wall_clock'] for line in n_iter_class]
            ax.semilogy(p_list,wall_clock_list,'-o')

    if fn=='energy':
        ax.axhline(y=-(E[1]-E[0])/E[0]) # Plot a horizontal line at the first excited state.
        ax.axhline(y=-(E[1]-E[0])/E[0]/2,ls='--') # Plot a horizontal dashed line halfway the ground state and the first excited state.
        ax.set_ylabel('Relative energy error')
    elif fn=='infidelity':
        ax.set_ylabel('Infidelity')
    elif fn=='wall_clock':
        ax.set_ylabel('Wall-clock time (h)')

    # On the x-axis, put the number of cycles rather then the number of parameters. 
    ax.set_xlabel('p') 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(n_iter_set, title='n_iter')
    plt.title(path)


    # Write to disk.
    if fn=='energy':
        plt.savefig(path+'/E_VQE.pdf')
    if fn=='infidelity':
        plt.savefig(path+'/inf_VQE.pdf')    
    if fn=='wall_clock':
        plt.savefig(path+'/wall_clock.pdf')
