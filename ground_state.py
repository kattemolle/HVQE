"""
Compute the energies of the k lowest energy eigenstates of the Heisenberg model defined on a given graph.

Parameters
----------
path : str
  Path to folder. The script looks for the file graph_input.txt in this folder. This file should define the list complete_graph_input=[...]. This is the graph on which the energy is computed. If the file also defines an anisotropy_input=float, the ground state is computed for the Heisenberg XXZ model with the given graph and anisotropy. 

k : str
  Number of lowest energy eigenstates of which the energy is to be computed. For k=1, the ground state energy is computed. 

Returns
-------
lowest_energies.txt : text file
  This script creates the file lowest_energies.txt in the folder that the script received as an argument. The file contains one of the k energies per line, in assending energy.

Examples
--------
If the GS energy of the Heisenberg model on a square is to be computed, save

complete_graph_input=[ ((0,0),(0,1)), ((0,1),(1,1)), ((1,1),(1,0)), ((1,0),(0,0)) ]

in a file named graph_input.txt. Put the file in ~/folder/

If you want to compute the two lowest energies of this model, run in a terminal

$ python3 compute_gs.py ~/square 2

--------------------------------------------------
Computing the 2 lowest energies of ~/square
The 2 lowest energies are [-8. -4.]
Solutions found using using ARPACK, in a total of 0.019322872161865234 seconds
--------------------------------------------------

The file gs_energies.txt will now be created, containing the two lowest energies of the Heisenberg model on a square. 

"""

import numpy as np
import qem
import sys
from time import time

return_state = True

# %% Read input from file
path = sys.argv[1]
k = int(sys.argv[2])

if path[-1] == "/":
    path = path[:-1]

with open(path + "/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
del complete_graph_input

# %% Run
print("")
print("--------------------------------------------------")
print("Computing the", k, "lowest energies of", path)
start = time()
output = qem.ground_state(complete_graph, k, return_state)
if qem.GPU == True:
    qem.sync()
end = time()
if return_state == False:
    print("The", k, "lowest energies are", output)
if return_state == True:
    print("The", k, "lowest energies are", output[0])
print("Solutions found using using ARPACK, in a total of", end - start, "seconds")
print("--------------------------------------------------")
print(" ")

## Write gs energy to disk
if return_state == False:
    np.savetxt(path + "/lowest_energies.txt", output)

if return_state == True:
    np.savetxt(path + "/lowest_energies.txt", output[0])

## Write gs itself to disk if return_state is True

if return_state == True:
    import pickle

    with open(path + "/gs.dat", "wb") as file:
        pickle.dump(output[1][:, 0], file, protocol=4)

## To read the ground states themselves, use
# with open('gs.dat','rb') as file:
#    st=pickle.load(file)
