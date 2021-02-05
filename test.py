import qem
import _HVQE
import pickle
import numpy as xp
import chainer as ch

# Load the ansatz from graph_input.txt
with open('data/kagome/periodic/own_graph/ops/energy/4x2/graph_input.txt', 'r') as file:
    exec(file.read())
complete_graph=complete_graph_input
init_layer=init_layer_input
layers =layers_input
del complete_graph_input
del init_layer_input
del layers_input

# Get the number of qubits from the complete_graph.
nodes=[node for edge in complete_graph for node in edge]
nodes=set(nodes)
n=len(nodes)
del nodes

# Load the true ground state into memory for computation of infidelities. 
gs_reg=qem.Reg(n)
with open('data/kagome/periodic/own_graph/ops/energy/4x2/gs.dat','rb') as file:
    gs_reg.psi.re=xp.array(pickle.load(file)).reshape((2,)*n)   

init_reg=qem.Reg(n)   
#for edge in init_layer:
#    qem.apply_prepare_singlet(edge,init_reg)

print(qem.infidelity(gs_reg,init_reg))

parameters=ch.Variable(xp.array([]))

inf=_HVQE.infidelity_from_parameters(init_reg,layers,n,1,parameters,gs_reg)
print(inf)

parameters=ch.Variable(xp.random.rand(96))

inf=_HVQE.infidelity_from_parameters(init_reg,layers,n,1,parameters,gs_reg)
print(inf)

