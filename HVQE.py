"""
(For help on how to run this script, run python3 HVQE.py -h.)

This script runs a VQE of the Heisenberg model on a graph that it receives as input. It also receives the ansatz for the VQE as input. 
Both the graph of the model and the ansatz are to be stored in a file graph_input.txt. 
This file should contain the following as seperate lines, in any order. 
---------------------------------------
complete_graph_input=list # List of edges of the form [(int,int),(int,int)] that determine the complete graph of the model. 
init_layer_input=list # List of edges on which singlet states are created as starting point. 
layers_input=list # List of layers. A layer contains edges. Every edge corresponds to gate in the anstaz of the VQE. If later more parameters than gates are specified, the layers in layers_input are repeated, starting again form the first layer.
---------------------------------------

Example
-------
If your files are stored in `path`, you want to use `n_par` parameters, where every parameter is repeated `par_multiplicity` times, run
``` 
python3 HVQE.py path n_par par_multiplicity
```
"""
import qem
import _HVQE
from time import time
import os
import chainer as ch
import pickle
from datetime import datetime

class Name:
    pass

# Get command line input
cmd_args=_HVQE.get_command_line_input()
run_args=Name()

try: # Use GPU if CuPy installation is available. 
    import cupy as xp
    run_args.GPU=True
except ImportError:
    import numpy as xp
    run_args.GPU=False

# Make timestamp in UTC of start
run_args.date_start=str(datetime.utcnow())

# Load the ansatz from graph_input.txt
with open(cmd_args.path+'/graph_input.txt', 'r') as file:
    exec(file.read())
run_args.complete_graph=complete_graph_input
run_args.init_layer=init_layer_input
run_args.layers =layers_input
del complete_graph_input
del init_layer_input
del layers_input

# Get the number of qubits from the complete_graph.
nodes=[node for edge in complete_graph for node in edge]
nodes=set(nodes)
run_args.n=len(nodes)
del nodes

# Load the true ground state into memory for computation of infidelities. 
gs_reg=qem.Reg(run_args.n)
with open(cmd_args.path+'/gs.dat','rb') as file:
    gs_reg.psi.re=xp.array(pickle.load(file)).reshape((2,)*run_args.n)   

# Print info about current run to stdout.    
print('Started basinhopping at',run_args.date_start, 'UTC, with'),
print('command line arguments =',vars(cmd_args))
print('runtime variables = ', vars(run_args))

# Prepare the init_reg, whose state is a dimer covering of run_args.complete_graph, as specified by run_args.init_layer.
init_reg=qem.Reg(run_args.n)   
for edge in run_args.init_layer:
    qem.apply_prepare_singlet(edge,init_reg)

###### RUN THE VQE #####
run_args.start=time()
vqe_out=_HVQE.run_VQE(cmd_args,run_args,init_reg)
if run_args.GPU==True:
    qem.sync()
run_args.end=time() 
run_args.wall_clock=(run_args.end-run_args.start)/60/60 # Wall-clock time of VQE run only, in hours. In the case of a restart, it measures the time from the beginning of the first run to the end of the last run, even if there were large times in between. 
########################

# Get the infidelity and the energy of the final state irrespective of whether we used the energy or the infidelity as the cost functon.
vqe_out.opt_parameters=ch.Variable(xp.array(vqe_out.opt_parameters))
if cmd_args.cost_fn=='energy':
    run_args.E_VQE=vqe_out.cost_VQE #Already a float
    run_args.inf_VQE=_HVQE.infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters,gs_reg)
    run_args.inf_VQE=float(run_args.inf_VQE.array)
if cmd_args.cost_fn=='infidelity':
    run_args.inf_VQE=vqe_out.cost_VQE #Already a float
    run_args.E_VQE=_HVQE.Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters)
    run_args.E_VQE=float(run_args.E_VQE.array)

vqe_out.opt_parameters=vqe_out.opt_parameters.array.tolist() #Convert for printing and storing.
run_args.date_end=str(datetime.utcnow()) # End time in UTC



# Write input and results to disk. If no former output exists, print a line explaining the data in the output file.

output=str([vars(cmd_args),vars(run_args),vars(vqe_out)])

if not os.path.exists(cmd_args.path+'/output.txt'):
    f=open(cmd_args.path+'/output.txt', 'w')
    f.write("### Output of the VQEs is written to this file. Note that the first local minimum is not included in the lists local_min_list,local_min_parameters_list and local_min_accept_list. See the comment at de function definition of 'callback' inside HVQE.py \n")
with open(cmd_args.path+'/output.txt', 'a') as f:
    f.write(output+'\n\n')

# Update plot of datapoints
gates_per_cycle=len([edge for layer in run_args.layers for edge in layer])
_HVQE.plot_VQE_data(cmd_args.path,'energy',cmd_args.par_multiplicity,gates_per_cycle)
_HVQE.plot_VQE_data(cmd_args.path,'infidelity',cmd_args.par_multiplicity,gates_per_cycle)
_HVQE.plot_VQE_data(cmd_args.path,'wall_clock',cmd_args.par_multiplicity,gates_per_cycle)

# Write input and results to stdout
print(' ')
print('==========================================================================================')
print('Finished basinhopping of ',cmd_args.path, 'at',run_args.date_end,'UTC, with')
print(' ')
print(vars(cmd_args))
print('init_par =', vqe_out.init_par)
print(' ')
print('RESULTS:')
print('--------')
print('E_VQE =',run_args.E_VQE)
print('inf_VQE =',run_args.inf_VQE)
print('n_fn_calls =',vqe_out.n_fn_calls)
print('Wall-clock time', run_args.wall_clock, 'hours')
print(' ')
print('For more output, see',cmd_args.path+'/output.txt')
print('Plots of output saved in',cmd_args.path)
print('End of HVQE.py')
print('==========================================================================================')



