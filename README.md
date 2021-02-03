# HeisenbergVQE
<img src="https://github.com/barbireau/HVQE/blob/main/images/logo.jpg" width="200"/>

A python package tailored to Variational Quantum Eigensolvers for the Heisenberg model, with optional GPU acceleration. Documentation, with plenty examples, is contained in the docstrings of the files outlined under [Files and folders](##files-and-folders).

Contents:<br>
[Example](#example)<br>
[Files and folders](#files-and-folders)<br>
[Installation](#installation)<br>


##Example: a square
Say you want to use a VQE to find the ground state of the Heisenberg model on a square. In this example I will assume you have installed HeisenbergVQE into ~/HVQE (also see [installation](#installation)). Create a text file with the contents

```python
complete_graph_input=[(0,1),(1,2),(2,3),(3,0)]
init_layer_input=[(0,1),(2,3)]
layers_input=[
	[(1,2),(3,0)],
	[(0,1),(2,3)]
	]
```

and save it under `~/HVQE/graph_input.txt`.

Here:

- The list of edges `complete_graph_input` defines the graph the Hamiltonian is defined on by defining its edges, using integers as vertices. 
- The list of edges `init_layer_input` defines the initial state of the register by specifying the edges along which singlets are to be created.
- The list of lists of edges `layers_input` defines the cycle of the ansatz. The n'th list contains the edges along which Heisenberg gates `e^(-i angle (XX+YY+ZZ)/4)` are to placed in the n'th layer of the cycle. Here, `angle` is the parameter of the gate, and `XX` stands for the tensor product of two Pauli-X operators (likewise for `YY`,`ZZ`).

To be able to asses the performance of the VQE, first compute the ground state and the first excited state by exact diagonalisation:

```bash
$ python3 ~/HVQE/ground_state.py ~/HVQE 2
```
(The script `~/HVQE/ground_state.py` looks for the file `graph_input.txt` inside the folder `~/HVQE`. It then computes the ground state of the `2` lowest energy eigenstates and stores results it in the folder `~/HVQE`)

Now we can run the VQE sequentially using 0 parameters (i.e. just return the energy of the init state) and 4 parameters, with a parameter multiplicity of 1 (i.e. every Heisenberg gate gets its own parameter), 

```bash
$ python3 ~/HVQE/HVQE.py ~/HVQE 0 1
$ python3 ~/HVQE/HVQE.py ~/HVQE 4 1
```
(For more info about the syntax, run `$ python3 ~/HVQE/HVQE.py -h`.)

This will save the following semi-log plot of the result under `~/HVQE/E_VQE.pdf`:

<img src="https://github.com/barbireau/HVQE/blob/main/images/E_VQE.jpg" width="400"/>

The vertical axis displays the relative error to the ground state `(E_VQE-E_0)/E_0`, with `E_VQE` the optimal energy found by the VQE and `E_0` the energy of the true ground state. The horizontal axis displays the number of cycles used in the ansatz. The solid horizontal line is at the energy of the first excited state. The dashed line is halfway between the first exited state and the ground state. (The ground state itself is not shown because it is at 0.)

The VQE has already found the exact ground state of the Heisenberg model on the square after a single cycle of the ansatz (a circuit of depth 2).

##Files and folders
For usage of HeisenbergVQE, the most important files and folders are:

- `data/`. Contains pre-produced data that was obtained by running `HVQE.py`. The folder contains a tree of folders, where every leaf contains all input and output pertaining to a single, specific system and ansatz.

- `ground_state.py`. Usage: `$ python3 ground_state.py path k`.
Compute the `k` lowest energy eigenstates of the Heisenberg model on the graph `complete_graph_input` which should be defined in `path/graph_input.txt`. The (flattened) ground state is saved under `path/gs.dat`, and a list of the `k` lowest energies is stored under `path/lowest_energies.txt`.

- `HVQE.py`. Run the VQE, calling standard emulator functions from `qem.py`. For info on command-line usage of `HVQE.py`, run `$ python3 HVQE.py -h`.

- `qem.py`. (qem is short for Quantum EMulator) This script defines standard functions and classes needed for the emulation of quantum circuits.

##Installation

- If you have git installed, clone the repo by `$ git clone https://github.com/barbireau/HVQE.git`. Alternatively, you can download and unzip this repo via the green 'code' button at the top right of the github page. 

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/). If you have a CUDA-enabled GPU and you want to use GPU acceleration, `cd` to your local repo and run `$ conda env create -f environment/environment_GPU.yml`. If you do not have such a GPU, or if you do not wish to use the GPU, change the command to `$ conda env create -f environment/environment_no_GPU.yml`.

- Test your installation by running `$ python3 test_HVQE.py`. This may take a minute.

- You're all set for running a VQE for your own graphs and ansätze!