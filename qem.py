try: # See if CuPy is installed. If false, continue without GPU.
    import cupy as xp
    print('CuPy installation found, continuing using GPU acceleration.')
    GPU=True
except ImportError:
    print('No CuPy installation found, continuing without GPU acceleration.')
    import numpy as xp
    GPU=False

import numpy #For the cases where GPU==True and we still want to use NumPy.
import chainer as ch
import scipy.sparse.linalg
import scipy.linalg
from copy import deepcopy
import os
from functools import reduce

try:
    os.sched_setaffinity(0,range(1000)) # Reset the CPU affinity; allow Python to use all CPUs that are available. 
    print('having the CPUs', os.sched_getaffinity(0), 'available.')
except AttributeError:
    pass

# Only use when timing: may slow down computation if used improperly 
if GPU==True:
    sync=xp.cuda.stream.get_current_stream().synchronize

# Classes

class Array:
    """
    Chainer does not natively support complex numbers. We need Chainer for automatic differentiation. This class defines a new type of array that stores the real part and imaginary part separately, as two real arrays. If an array is purely real (imaginary) the imaginary part (real part) should be stored as an array of the same shape containing zeros. The real and imaginary part should be either NumPy/CuPy arrays or `chainer.Variable`s. 

    Attributes
    ----------
    re : chainer.Variable, numpy.ndarray or cupy.ndarray

    im : chainer.Variable, numpy.ndarray or cupy.ndarray

    shape : tuple

    """
    def __init__(self,re,im):
        assert re.shape==im.shape, 'Array.re and array.im must have same shape.'
        self.re=re
        self.im=im
        self.shape=self.re.shape
            
    def __str__(self):
        return str(self.re)+'\n + i* \n'+str(self.im)

    def __add__(self, other):
        return Array(self.re+other.re,self.im+other.im)

    def __sub__(self,other):
        return Array(self.re-other.re,self.im-other.im)
    
    def __mul__(self,other): # Element-wise product of two objects of type Array.
        return Array(self.re*other.re-self.im*other.im, self.re*other.im+self.im*other.re)
    
    def dagger(self): # Return daggered array.
        return Array(self.re,-self.im)

    def reshape(self,shape): # Not this always creates a copy, so use with care.
        return Array(self.re.reshape(shape),self.im.reshape(shape))
            
    def do_dagger(self): # Apply dagger, return None
        self.im=-self.im


class Reg:
    """
    The quantum register class. Holds the current wave function of the register as a qem.Array array of shape (2,)*n. Upon creation, the wave function is initialized as the wave function of |00...0>.

    Parameters
    ----------
    n : int
      The number of qubits of the register.

    Attributes
    ----------
    n : int
      The number of qubits of the register.

    psi : qem.Array 
      The wave function of the register. A quantum state $\sum_{ab} \psi_{ab...} |ab...>$ is stored in such a way that psi[a,b,...] holds value $\psi_{ab...}$. Is initialized to the wavefunction of |00...0>

    Methods
    -------
    reset()
        Reset the state of the reg to |00...0>.

    print_ket_state()
        Print the state of the reg to stdout in ket notation, in e.g. the form (0.707107+0j)|00> + (0.707107+0j)|11>.

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> print(reg.psi)
    [[1. 0.]
     [0. 0.]]
     + i* 
    [[0. 0.]
     [0. 0.]]
    >>> reg.print_ket_state()
    psi = (1+0j)|00>
    """
    def __init__(self,n):
        self.n=n
        self.psi=xp.zeros((2,)*self.n,dtype=xp.float64)
        self.psi[(0,)*self.n]=1.0
        self.psi=Array(self.psi, xp.zeros((2,)*self.n,dtype=xp.float64))

    def reset(self):
        """
        Reset the wavefunction to that of |00..0>.
        """
        del self.psi
        self.psi=xp.zeros((2,)*self.n,dtype=xp.float64)
        self.psi[(0,)*self.n]=1.0
        self.psi=Array(self.psi, xp.zeros((2,)*self.n,dtype=xp.float64))

    def print_ket_state(self):
        get_bin=lambda x: format(x, 'b').zfill(self.n) # Def function that converts ints to the binary representation and outputs it as string.
        re=self.psi.re
        im=self.psi.im
        if type(re)==ch.Variable:
            re=re.data
        if type(im)==ch.Variable:
            im=im.data        
        psi=re+1j*im
        psi=psi.flatten()
        psi=numpy.around(psi,6)
        out='psi = '
        for i in range(len(psi)):
            if psi[i]!=0:
                out+=str(psi[i])+'|'+get_bin(i)+'>'+' + '
        out = out[:-3]
        print(out)
        
class EmptyReg:
    """
    Same as Reg but initializes without data arrays for the state qem.Reg.psi.  
    """
    def __init__(self,n):
        self.n=n
        self.psi=Array(xp.array([]),xp.array([]))

    def reset(self):
        """
        Resets the wavefunction to that of |00..0>. No parameters.
        """
        del self.psi
        self.psi=Array(xp.array([]),xp.array([]))

class Gate:
    """
    A gate is implemented as an object. The parameter can be either a qem.Array of shape (2**m,2**m) or (2,)*m, where m is the number of qubits the gate acts upon. In converting matrices to tensors of shape (2,)*m, we use the convention that [000...,000...] refers to the upper right entry of the matrix. In the tensor format, this correspronds to the entry [0,0,0,...,0,0,0,...].  

    Parameters
    ----------
    array : qem.Array of shape (2**m,2**m) or (2,)*m with m the number of qubits the gate acts upon.

    Attributes
    ----------
    self.array: qem.Array
        The input qem.Array, but then qem.Array.re and qem.Array.im are reshaped to shape (2,)*m.

    Methods:
    --------
    get_matrix() : return the array but reshaped to (2**m,2**m), irrespective of the shape that was used to construct the object.
    """
    def __init__(self,array):
        assert type(array)==Array
        self.array=array
        if array.shape!=(2,2,2,2) and array.shape!=(2,2):
            totaldim=reduce((lambda x, y: x * y), array.shape)
            self.array=array.reshape((2,)*int(numpy.log2(totaldim)))
    def get_matrix(self):
        return self.array.reshape((2**(len(self.array.shape)//2),2**(len(self.array.shape)//2)))

    # Predefined gates that are children of the Gates with preset matrices
 
class X(Gate):
    array=Array(xp.array([[0.,1.],[1.,0.]]),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='X'

class Y(Gate):
    array=Array(xp.array([[0.,0.],[0.,0.]]),xp.array([[0.,-1.],[1.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='Y'

class Z(Gate):
    array=Array(xp.array([[1.,0.],[0.,-1.]]),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='Z'

class H(Gate):
    array=Array(xp.array([[1.,1.],[1.,-1.]]/numpy.sqrt(2)),xp.array([[0.,0.],[0.,0.]]))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='H'

class CNOT(Gate):
    array=Array(xp.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]]),xp.zeros((4,4),dtype=xp.float64))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='CNOT'

class SWAP(Gate):
    array=Array(xp.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]),xp.zeros((4,4),dtype=xp.float64))
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='SWAP'

class Heisenberg():
    """
    The Heisenberg gate, with matrix (XX + YY + ZZ)/4. Gets special treatment in run_VQE.
    """
    pass
        
class Heisenberg_exp():
    """
    This is not like the other gates for reasons of speed. It is treated differently in `qem.apply_action`.

    Parameters
    ----------
    angle : chainer.Variable
    """
    def __init__(self,angle):
        self.angle=angle

class prepare_singlet(Gate):
    """
    This is a non-unitary operation that prepares the two-body ground state of (XX+YY+ZZ)/4, the singlet state, if acted on qubits in the state |00>.
    """
    array=Array( xp.array([[0,0,0,0],[1,0,0,0],[-1,0,0,0],[0,0,0,0]])/xp.sqrt(2) , xp.zeros((4,4)) )
    def __init__(self):
        Gate.__init__(self,self.array)
        self.name='prepare_singlet'
        
class Action():
    """ 
    An Action is a combination of a `qem.Gate' and the qubit ints this gate should act upon. Creation of an Action object does not apply the gate to the qubit. To do this, use `qem.apply_action()`. 

    Parameters
    ----------
    qubits : tuple of ints
        The qubits the gate should act upon. This is a tuple of one int for single-qubit gates, and a tuple of two ints for tow-qubit gates. 

    gate : qem.Gate
    """
    def __init__(self, qubits,gate):
        self.qubits=qubits
        self.gate=gate

class Layer():
    """
    A layer is a list of actions (`qem.action`). We imagine all the actions in a layer would be excequted simultaniously on a quantum computer. This behaviour is, however, not enforced, and a layer can have multiple actions acting on the same qubit, but in that case care should be taken to ensure that gates are applied in the right order.
    
    Parameters
    ----------
    *args : Arguments
        - None 
          self.actions is initialized as an empty list. 

        - list : list
          List of actions (`qem.action`).

    Methods
    -------
        append_action(action)
            Append the action to the layer.
    """
    def __init__(self,*args):
        if len(args)==0:
            self.actions=[]
        if len(args)==1:
            self.actions=args[0]

    def append_action(self, action):
        self.actions.append(action)

class Cir():
    """
    A circuit is has list of layers (qem.layer). Every layer is a list of actions (qem.action). Every action has a qubits list and a gate (qem.gate). 
    The circuit is initialized with as having one layer with no actions. 

    Attributes
    ----------
    layers : list
        List containing layers (`qem.layer`). 
    
    Methods
    -------
    append_layer(*args)
        append a new layer to self.layers. If no args are given, the new Layer is empty. If one argument is given, it should be a list of actions. Then this list of actions is the new layer. 
    
    append_action(action) 
        Append an Action to the last layer.

    Examples
    --------
    
    >>> import qem
    >>> cir=qem.Cir() # Create circuit with one empty layer
    >>> cir.append_action(qem.Action((1),qem.H())) # Append a Hadamard action to the layer. 
    >>> cir.append_layer() # Append a new empty layer to the circuit.
    >>> cir.append_action(qem.Action((0,1),qem.CNOT())) # Append a CNOT action to the new emtpy layer.

    The circuit now contains the layers
    >>> print(cir.layers)
    [<qem.Layer object at 0x101a163950>, <qem.Layer object at 0x1018f9abd0>]

    Of which the 0th layer contains the actions
    >>>print(cir.layers[0].actions)
    [<qem.Action object at 0x101a1638d0>]

    Of which the 0th action has the gate with name
    >>> print(cir.layers[0].actions[0].gate.name)
    H
    """
    def __init__(self):
        self.layers=[Layer()]
      
    def append_layer(self,*args):
      if len(args)==0:
          self.layers.append(Layer())
      elif len(args)==1:
          assert type(args[0])==list, 'Argument must be a list of actions.'
          self.layers.append(args[0])
      else:
          raise Exception('Zero or one arguments expected but recieved '+str(len(args)))
  
    def append_action(self,action):
        self.layers[-1].append_action(action)

# Functions

def round_assert(a,b):
    """
    Assert the xp arrays a and b are equal up to 4 decimal places.
    """
    return xp.testing.assert_array_equal(xp.around(a,decimals=5),xp.around(b,decimals=5))


def tensordot(a,b,c):
    """
    Return the tensordot of two objects from the class `Array`. Also see `numpy.tensordot`.
    """
    assert type(a)==type(b)==Array
    re=ch.functions.tensordot(a.re,b.re,c)-ch.functions.tensordot(a.im,b.im,c)
    im=ch.functions.tensordot(a.re,b.im,c)+ch.functions.tensordot(a.im,b.re,c)
    return Array(re,im)

def moveaxis(a,b,c):
    """
    Of the `qem.Array' a, move axis a to position b. Also see `numpy.moveaxis`.
    """
    re=ch.functions.moveaxis(a.re,b,c)
    im=ch.functions.moveaxis(a.im,b,c)
    return Array(re,im)
  
def apply_action(action, reg):
    """
    Applies the action to the register reg, thereby chainging the resister's state (reg.psi). 

    Parameters
    ----------
    action : qem.Action

    reg : qem.Reg

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> action=qem.Action((0),qem.H())
    >>> qem.apply_action(action,reg)
    >>> action=qem.Action((0,1),qem.CNOT())
    >>> qem.apply_action(action,reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.         0.70710678]])
     + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    # qem.Heisenberg_exp gets a special treatment. This way it is faster. Note e^(-i angle/4) * e^(-i angle (XX+YY+ZZ)/4)=cos(angle/2) Id - i sin(angle/2) SWAP.
    if type(action.gate)==Heisenberg_exp:
        angle=action.gate.angle
        reg_id=EmptyReg(reg.n)
        reg_SWAP=EmptyReg(reg.n)

        reg_id.psi.re=ch.functions.cos(angle/2.)*reg.psi.re
        reg_id.psi.im=ch.functions.cos(angle/2.)*reg.psi.im

        # Multiply SWAP term with sin
        reg_SWAP.psi.re=ch.functions.sin(angle/2.)*reg.psi.re
        reg_SWAP.psi.im=ch.functions.sin(angle/2.)*reg.psi.im

        # Multiply SWAP term with -i
        c=reg_SWAP.psi.re 
        reg_SWAP.psi.re=reg_SWAP.psi.im
        reg_SWAP.psi.im=-c

        # Do the SWAP
        reg_SWAP.psi.re=ch.functions.swapaxes(reg_SWAP.psi.re,*action.qubits)
        reg_SWAP.psi.im=ch.functions.swapaxes(reg_SWAP.psi.im,*action.qubits)

        # Add the SWAP term to the identity term
        reg.psi=reg_id.psi+reg_SWAP.psi

    # Also the gate Heisenberg() gets special treatment, very much like Heisenberg_exp. Note (XX+YY+ZZ)/4=SWAP/2-Id/4.
    elif type(action.gate)==Heisenberg:
        reg_id=EmptyReg(reg.n)
        reg_SWAP=EmptyReg(reg.n)

        reg_id.psi.re=-reg.psi.re/4
        reg_id.psi.im=-reg.psi.im/4

        reg_SWAP.psi.re=reg.psi.re/2
        reg_SWAP.psi.im=reg.psi.im/2

        reg_SWAP.psi.re=ch.functions.swapaxes(reg_SWAP.psi.re,*action.qubits)
        reg_SWAP.psi.im=ch.functions.swapaxes(reg_SWAP.psi.im,*action.qubits)

        # Add the SWAP term to the identity term
        reg.psi=reg_id.psi+reg_SWAP.psi
        
    else:
        n_legs=len(action.gate.array.shape)
        lower_legs=range(n_legs//2,n_legs)
        reg.psi=tensordot(action.gate.array,reg.psi,(lower_legs,action.qubits))
        reg.psi=moveaxis(reg.psi,range(n_legs//2),action.qubits)

def run(cir,reg):
    """
    Run the circuit cir on the register reg, thereby changing the quantum state of the register. 

    Paramters
    ---------
    cir : qem.Circuit
    reg : qem.Reg

    Examples
    --------
    Create a GHZ state on 8 qubits.
    >>> import qem
    >>> reg=qem.Reg(8)
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action((0),qem.H()))
    >>> for i in range(7):
    ...     cir.append_layer()
    ...     cir.append_action(qem.Action((i,i+1),qem.CNOT()))
    >>> qem.run(cir,reg)
    >>> reg.print_ket_state()
    psi = (0.707107+0j)|00000000> + (0.707107+0j)|11111111>
    """                           
    for layer in cir.layers:
        for action in layer.actions:
            apply_action(action,reg)
        
def ground_state(g,k,return_state=False):
    """
    Compute the k lowest energies of the Heisenberg model defined on the graph g. (If k=1 only the ground state energy is computed.) The nodes of the graph need not be integers or coordinates (as is the case for test_graph_input.edges_fig() and related functions). If return_state=True, also the whole state vector is returned. 

    Optionally, a 'weight' attribute can be set for edges, which we will call w_e here. Then the Hamiltonain will read \sum_e w_e (X_e1 X_e2 + Y_e1 Y_e2 + Z_e1 Z_e2)/4. w_e defaults to 1 for the edges where no weight is given. 
    
    Parameters
    ----------
    g : list
        The graph on which the Heisenberg model is defined as a list of edges, where every edge is of the form (int,int).

    k : int
        The energy is computed of the k states with the lowest energy. For k=1 only the ground state energy is computed.

    return_state : Bool (optional)
        If true, also the whole state vector is returned.

    Returns
    -------
    w : numpy.ndarray (dtype=numpy.float64)
        Array containing the k lowest eigenvalues in increasing order.
        If return_state==True, also the state vectors are returned. Then the output is equal to that of scipy.linalg.eighs. This means in this case w=[b,v] with b the array containing the k lowest eigenvalues, and v an array representing the k eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i]. Note the ground state is retrned as a flat array (as opposed to the shape of e.g. qem.Reg.psi and functions as qem.basis_state()).
    

    Notes
    -----
    This function uses a Lanszos algorithm (ARPACK via scipy.sparse.linalg.eigsh) to compute the energy memory-efficiently. The storage of the complete Hamiltonian, even as a sparse matrix, can be very costly. Therefore, the Hamiltonian is suplied to scipy.linalg.eighs as a callable. That is, a function that receives the vector r and returns H.r (the Hamiltonian applied to r).
    
    """
    heisenberg_tensor_real=xp.array([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]],dtype=xp.float64)/4
    heisenberg_tensor_real=heisenberg_tensor_real.reshape((2,2,2,2)) # Note heisenberg tensor is real. So also the Hamiltonian, and any vector during the Lanszos algo will be real.

    nodes=[node for edge in g for node in edge]
    nodes=set(nodes)
    n=len(nodes)
    del nodes
    
    def Hv(v):
        v=xp.array(v,dtype=xp.float64)
        v=v.reshape((2,)*n)
        vp=xp.zeros((2,)*n,dtype=xp.float64)
        for edge in g:
            new_term=xp.tensordot(heisenberg_tensor_real,v,((2,3),edge))
            new_term=xp.moveaxis(new_term,(0,1),edge)
            vp+=new_term
        vp=vp.flatten()
        if GPU==True:
            vp=xp.asnumpy(vp)
        return vp

    H=scipy.sparse.linalg.LinearOperator((2**n,2**n),matvec=Hv)
    output=scipy.sparse.linalg.eigsh(H,k,which='SA',maxiter=numpy.iinfo(numpy.int32).max)
    if return_state==False:
        return output[0]
    else:
        return output

# Gates as functions

def apply_prepare_singlet(qubits,reg):
    action=Action(qubits, prepare_singlet())
    apply_action(action,reg)

def apply_H(qubits,reg):
    """
    Apply the Hadamard gate to control (int) of the register.

    Parameters
    ----------
    qubits : tuple containing one int
        The number of the qubit the gate is to be applied to
    
    reg : qem.reg
        The register the gate H is to be applied to.
    
    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H(0,reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.70710678 0.        ]])
    + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    action=Action(qubits, H())
    apply_action(action,reg)

def apply_X(qubits,reg):
    """
    Apply the X gate to reg. See `qem.apply_H`.
    """
    action=Action(qubits, X())
    apply_action(action,reg)

def apply_Y(qubits,reg):
    """
    Apply the Y gate to qubits reg. See `qem.apply_H`.
    """
    action=Action(qubits, Y())
    apply_action(action,reg)

def apply_Z(qubits,reg):
    """
    Apply the Z gate to reg. See `qem.apply_H`.
    """
    action=Action(qubits, Z())
    apply_action(action,reg)

def apply_CNOT(qubits,reg):
    """
    Apply the CNOT gate to reg. Qubits is a tuple of the form (int,int), containing the control and target qubit number (in that order).
    ----------
    qubits : tuple (int,int)
        Tuple containing the control and target qubit numner (in that order).

    reg : qem.reg
        The register the CNOT is to be applied to.

    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H((0),reg)
    >>> qem.apply_CNOT((0,1),reg)
    >>> print(reg.psi)
    variable([[0.70710678 0.        ]
              [0.         0.70710678]])
    + i* 
    variable([[0. 0.]
              [0. 0.]])
    """
    action=Action(qubits, CNOT())
    apply_action(action,reg)

def Heisenberg_energy(g,reg):
    """
    Compute < reg.psi | H | reg.psi >, with H the Heisenberg Hamiltonian defined on the graph g.

    Parameters
    ----------
    g : list of edges or networkx.Graph 
        List of edges of the form (int,int) that define the graph. If it is a networkx.Graph object, this graph should already be mapped to ints (containing the 'old' node attribute completeness, but not required). In that case, the edges can additionally specify an edge attribute called 'weight'. This means that for this edge, the Hamiltonian is weight*(XX+YY+ZZ)/4.

    reg : qem.reg
        The resister containing the state for which the expectation value of the Hamiltonian is to be computed.   

    Retruns
    -------
    energy : chainer.Variable

    Example
    -------
    Compute the expectation value of the energy of the Neel state |0101> on a square. 
    >>> import qem
    >>> import numpy as np
    >>> edges=[(0,1),(1,2),(2,3),(3,0)]
    >>> reg=qem.Reg(4)
    >>> qem.apply_X(1,reg)
    >>> qem.apply_X(3,reg)
    >>> print(qem.Heisenberg_energy(edges,reg))
    variable(-1.)

    Compare this to the ground state energy of the Heisenberg model on the square.
    >>> qem.ground_state(edges,1)
    >>> print(qem.ground_state(g,1)[0].round())
    -2.0
    """
    E=0.
    reg_prime=EmptyReg(reg.n)
    gate=Heisenberg()
    for edge in g:
        reg_prime.psi=reg.psi 
        action=Action(edge,gate)
        apply_action(action,reg_prime)
        reg_prime.psi.do_dagger()
        E_term=tensordot(reg_prime.psi,reg.psi, (range(reg.n),range(reg.n)))
        E+=E_term.re
          
    return E

def compute_s(reg):
    """
    Checks if the state of the register is an eigenstate of the total spin operator S^2. If it is not an eigenstate, it returns None. If it is an eigenstate, it returns the quantum number S, defined by S^2|psi>=s(s+1)|psi>, with |psi> the eigenstate. 
    """
    #Check that the norm of the state of the register is unity
    norm=tensordot(reg.psi.dagger().flatten(),reg.psi.flatten(),((0),(0)))
    norm=xp.sqrt(norm.re.array**2+norm.im.array**2)
    assert xp.around(norm,5)==1., 'State of the register is not normalized.'
    reg_prime=Reg(reg.n)
    reg_prime.psi=Array(xp.zeros(reg.psi.shape), xp.zeros(reg.psi.shape))

    for i in range(reg.n):
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_X(j,reg_prime_prime)
            apply_X(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    for i in range(reg.n):        
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_Y(j,reg_prime_prime)
            apply_Y(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    for i in range(reg.n):
        for j in range(reg.n):
            reg_prime_prime=deepcopy(reg)
            apply_Z(j,reg_prime_prime)
            apply_Z(i,reg_prime_prime)
            reg_prime.psi=reg_prime.psi + reg_prime_prime.psi

    inner=tensordot(reg.psi.dagger().flatten(),reg_prime.psi.flatten(),((0),(0)))
    norm=tensordot(reg_prime.psi.dagger().flatten(),reg_prime.psi.flatten(),((0),(0)))
    norm=xp.sqrt(norm.re.array**2+norm.im.array**2)
    if xp.around(norm,5)==0.:
        print('State of register is eigenstate of the total spin operator, with s=0')
        return 0.
    elif xp.around(xp.sqrt(inner.re.array**2+inner.im.array**2)/norm,5)!=1.:
        print('State of register is not an eigenstate of the total spin operator')
        return None
    elif xp.around(xp.sqrt(inner.re.array**2+inner.im.array**2)/norm,5)==1.:
        print('State of register is eigenstate of the total spin operator, with')
        s=-1/2+1/2*xp.sqrt(1+4*norm)
        print('s=',s)
        return s
    else:
        raise ValueError()

def expectation(cir,reg):
    """
    Returns the expectation value <psi|U|psi>, where psi is the state of the register (=reg.psi) and U is the unitary induced by the circuit.
    
    Parameters
    ----------
    cir : qem.Circuit
      
    reg : qem.Reg

    Returns
    -------
    ex : qem.Array (with qem.re a chainer.Variable with shape (), likewise for qem.im)
    

    Examples
    --------
    Compute the expectation value <psi|Z_0 Z_1|psi> with |psi>=|0000>:
    >>> import qem
    >>> reg=qem.Reg(4)
    >>> cir=qem.Cir()
    >>> cir.append_action(qem.Action((0),qem.Z()))
    >>> cir.append_action(qem.Action((1),qem.Z()))
    >>> print(qem.expectation(cir,reg))
    variable(1.)
    + i* 
    variable(0.)
    """
    reg_prime=Reg(reg.n)
    reg_prime.psi=reg.psi
    run(cir,reg_prime)
    reg_prime.psi.do_dagger()
    ex=tensordot(reg.psi, reg_prime.psi,(range(reg.n),range(reg.n)))
    return ex

def spin_spin_correlation(i,j,reg):
    """
    Calculates the spin_spin correlation ( < S_i . S_j > - < S_i > . < S_j > ), with S=(X,Y,Z)/2.

    Parameters
    ----------
    i : int
        Number of the first spin.

    j : int
        Number of the second spin.

    reg : qem.Reg
        The quantum register hosting the spins i and j.

    Returns 
    -------
    c : qem.Array
        The value of the correlation as a qem.Array object (with c.re a chainer.Variable of shape () and likewise for c.im .)
    
    Examples
    --------
    >>> import qem
    >>> reg=qem.Reg(2)
    >>> qem.apply_H((0),reg)
    >>> qem.apply_CNOT((0,1),reg)
    >>> cor=qem.spin_spin_correlation(0,1,reg)
    >>> print(cor)
    variable(0.25)
    + i* 
    variable(0.)
    """
    #  ( < S_i . S_j > - < S_i > . < S_j > ) = 1/4( <X_i X_j> + <Y_i Y_j> + <Z_i Z_j> - <X_i><X_j> - <Y_i><Y_j> - <Z_i><Z_j> )
    def double(i,j,gate):
        cir=Cir()
        cir.append_action(Action((i),gate))
        cir.append_action(Action((j),gate))
        return expectation(cir,reg)

    def single(i,j,gate):
        cir=Cir()
        cir.append_action(Action((i),gate))
        return expectation(cir,reg)*expectation(cir,reg)

    c=Array(xp.array(1/4.),xp.array(0.)) * ( double(i,j,X()) + double(i,j,Y()) + double(i,j,Z()) - single(i,j,X()) - single(i,j,Y()) - single(i,j,Z()) )

    return c

def infidelity(reg,reg_prime):
    """
    Computes the infidelity 1- |< reg.psi | reg_prime.psi >|^2  
    
    Returns
    -------
    inf : chainer.Variable with cupy array of shape ().
    """
    inner=tensordot(reg.psi.dagger(),reg_prime.psi,(range(reg.n),range(reg.n))) # Returns a qem.Array filled with ch.Variable s with cupy data shape ().
    inner_sq=inner*inner.dagger() # Same
    inf=1-inner_sq.re # The infidelity as 1 minus the SQUARED overlap. Cost is now a chainer Variable of shape ()
    return inf
