"""
This script contains tests for HVQE.
"""
try:  # See if CuPy is installed. If false, continue without GPU.
    import cupy as xp

    GPU = True
except ImportError:
    import numpy as xp

    GPU = False

import numpy as numpy
import qem
import _HVQE
import chainer as ch
import pickle

print("Testing HVQE")


# gs chain 6

with open("data/chain/periodic/own_graph/ops/energy/6/graph_input.txt", "r") as file:
    exec(file.read())


complete_graph = complete_graph_input
layers = layers_input
init_layer = init_layer_input


nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_en = qem.ground_state(complete_graph, 1)[0]

assert numpy.round(gs_en, 6) == numpy.round(
    -2.80277564, 6
)  # RHS is calculated in test_HVQE.nb


# gs kagome 2x2

complete_graph_input = None
layers_input = None
init_layer_input = None
with open("data/kagome/periodic/own_graph/ops/energy/2x2/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
layers = layers_input
init_layer = init_layer_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_en = qem.ground_state(complete_graph, 1)[0]

assert numpy.round(gs_en, 6) == numpy.round(-5.44487522, 6)  # Idem


# test_gs_kagome3x2

complete_graph_input = None
layers_input = None
init_layer_input = None
with open("data/kagome/periodic/own_graph/ops/energy/3x2/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
layers = layers_input
init_layer = init_layer_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_en = qem.ground_state(complete_graph, 1)[0]

assert numpy.round(gs_en, 6) == numpy.round(-8.04827077, 6)  # Idem

# test_chain6():
complete_graph_input = None
layers_input = None
init_layer_input = None
par_multiplicity = 1
parameters = ch.Variable(
    xp.array(
        [
            0.5338210879053804,
            -4.861452010173478,
            3.749724997117319,
            1.027085088315441,
            -0.9003289468166109,
            1.2980981922608912,
            4.694974774263656,
            3.346625458188054,
            2.3291749603764416,
        ]
    )
)

with open("data/chain/periodic/own_graph/ops/energy/6/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
init_layer = init_layer_input
layers = layers_input
del complete_graph_input
del init_layer_input
del layers_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_reg = qem.Reg(n)
with open("data/chain/periodic/own_graph/ops/energy/6/gs.dat", "rb") as file:
    gs_reg.psi.re = xp.array(pickle.load(file)).reshape((2,) * n)

init_reg = qem.Reg(n)
for edge in init_layer:
    qem.apply_prepare_singlet(edge, init_reg)

E = _HVQE.Heisenberg_energy_from_parameters(
    complete_graph, init_reg, layers, n, par_multiplicity, parameters
)
E.backward()
grad_E = parameters.grad
E = float(E.array)

parameters = ch.Variable(
    xp.array(
        [
            0.5338210879053804,
            -4.861452010173478,
            3.749724997117319,
            1.027085088315441,
            -0.9003289468166109,
            1.2980981922608912,
            4.694974774263656,
            3.346625458188054,
            2.3291749603764416,
        ]
    )
)

inf = _HVQE.infidelity_from_parameters(
    init_reg, layers, n, par_multiplicity, parameters, gs_reg
)
inf.backward()
grad_inf = parameters.grad
inf = inf.array

assert numpy.round(E, 6) == numpy.round(-1.55200433, 6)
qem.round_assert(
    grad_E,
    [
        -0.05005307,
        -0.45674392,
        -0.19267117,
        0.09687651,
        -0.27705326,
        -0.13248857,
        0.28196716,
        0.05205173,
        -0.56275853,
    ],
)

assert numpy.round(inf, 6) == numpy.round(0.58861091, 6)
qem.round_assert(
    grad_inf,
    [
        1.42404631e-02,
        -1.00988019e-01,
        -1.85871448e-01,
        1.10099342e-01,
        3.24262071e-02,
        3.55959076e-05,
        1.72662807e-01,
        9.54738481e-02,
        -9.31588180e-02,
    ],
)

# test kagome 2x2 opg
complete_graph_input = None
layers_input = None
init_layer_input = None
par_multiplicity = 1
parameters = ch.Variable(
    xp.array(
        [
            -1.4590336907024906,
            5.1676688121457985,
            2.8216029812864107,
            -0.6880805233230696,
            -0.3164463289812076,
            -3.7855079246553895,
            -3.879886900967808,
            -0.7872929911255397,
            3.163342822710785,
            -0.13159909503576017,
            -1.7580940899494735,
            -5.900346380330523,
            -0.10797123542461406,
            4.86518613798269,
            0.917307768531554,
            1.683462483531322,
            -0.17943326392556003,
            -3.444665304255345,
        ]
    )
)

with open("data/kagome/periodic/own_graph/ops/energy/2x2/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
init_layer = init_layer_input
layers = layers_input
del complete_graph_input
del init_layer_input
del layers_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_reg = qem.Reg(n)
with open("data/kagome/periodic/own_graph/ops/energy/2x2/gs.dat", "rb") as file:
    gs_reg.psi.re = xp.array(pickle.load(file)).reshape((2,) * n)

init_reg = qem.Reg(n)
for edge in init_layer:
    qem.apply_prepare_singlet(edge, init_reg)

E = _HVQE.Heisenberg_energy_from_parameters(
    complete_graph, init_reg, layers, n, par_multiplicity, parameters
)
E.backward()
grad_E = parameters.grad
E = float(E.array)

parameters = ch.Variable(
    xp.array(
        [
            -1.4590336907024906,
            5.1676688121457985,
            2.8216029812864107,
            -0.6880805233230696,
            -0.3164463289812076,
            -3.7855079246553895,
            -3.879886900967808,
            -0.7872929911255397,
            3.163342822710785,
            -0.13159909503576017,
            -1.7580940899494735,
            -5.900346380330523,
            -0.10797123542461406,
            4.86518613798269,
            0.917307768531554,
            1.683462483531322,
            -0.17943326392556003,
            -3.444665304255345,
        ]
    )
)

inf = _HVQE.infidelity_from_parameters(
    init_reg, layers, n, par_multiplicity, parameters, gs_reg
)
inf.backward()
grad_inf = parameters.grad
inf = inf.array

assert numpy.round(E, 6) == numpy.round(-0.92843688, 6)
qem.round_assert(
    grad_E,
    [
        2.04644322e-01,
        -1.85569413e-01,
        2.46981435e-03,
        2.20623413e-01,
        -4.43852995e-01,
        1.88123420e-01,
        1.54393821e-01,
        -1.25588736e-01,
        1.12063086e-01,
        7.33843670e-02,
        2.54297895e-01,
        1.12838749e-01,
        -3.32148196e-04,
        1.29190901e-01,
        -1.10032142e-01,
        4.47627573e-01,
        5.18804549e-02,
        -8.73067841e-03,
    ],
)

assert numpy.round(inf, 6) == numpy.round(0.99028414, 6)
qem.round_assert(
    grad_inf,
    [
        -0.00276295,
        0.00047363,
        -0.00536181,
        0.00543681,
        -0.00161217,
        -0.00590244,
        0.0047045,
        0.00695224,
        -0.00433547,
        0.00531215,
        0.00109434,
        0.00244617,
        0.00066088,
        -0.0029838,
        0.00281999,
        0.00252471,
        0.00731822,
        -0.00318952,
    ],
)


# Kagome 2x2 ops
par_multiplicity = 2
parameters = ch.Variable(
    xp.array(
        [
            0.049431159925426016,
            2.0971826881228175,
            1.9558443268702845,
            -3.879886900967808,
            -0.7872929911255397,
            3.163342822710785,
            -0.10797123542461406,
            4.86518613798269,
            0.917307768531554,
            -5.5444502641063185,
            6.266268104834758,
            4.7765043994641445,
            3.59062779362851,
            5.276025913422831,
            4.927755050742828,
            4.888912700553572,
            0.1315880185269016,
            2.2552263123593193,
            -1.306859377781489,
            -0.9743835390864923,
            -2.905148883656384,
            -2.6995002982137706,
            -4.4101684202100255,
            -1.3593739583418731,
        ]
    )
)

with open("data/kagome/periodic/own_graph/ops/energy/2x2/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
init_layer = init_layer_input
layers = layers_input
del complete_graph_input
del init_layer_input
del layers_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_reg = qem.Reg(n)
with open("data/kagome/periodic/own_graph/ops/energy/2x2/gs.dat", "rb") as file:
    gs_reg.psi.re = xp.array(pickle.load(file)).reshape((2,) * n)

init_reg = qem.Reg(n)
for edge in init_layer:
    qem.apply_prepare_singlet(edge, init_reg)

E = _HVQE.Heisenberg_energy_from_parameters(
    complete_graph, init_reg, layers, n, par_multiplicity, parameters
)
E.backward()
grad_E = parameters.grad
E = float(E.array)

parameters = ch.Variable(
    xp.array(
        [
            0.049431159925426016,
            2.0971826881228175,
            1.9558443268702845,
            -3.879886900967808,
            -0.7872929911255397,
            3.163342822710785,
            -0.10797123542461406,
            4.86518613798269,
            0.917307768531554,
            -5.5444502641063185,
            6.266268104834758,
            4.7765043994641445,
            3.59062779362851,
            5.276025913422831,
            4.927755050742828,
            4.888912700553572,
            0.1315880185269016,
            2.2552263123593193,
            -1.306859377781489,
            -0.9743835390864923,
            -2.905148883656384,
            -2.6995002982137706,
            -4.4101684202100255,
            -1.3593739583418731,
        ]
    )
)

inf = _HVQE.infidelity_from_parameters(
    init_reg, layers, n, par_multiplicity, parameters, gs_reg
)
inf.backward()
grad_inf = parameters.grad
inf = inf.array

assert numpy.round(E, 6) == numpy.round(-1.51566624, 6)
qem.round_assert(
    grad_E,
    [
        -0.34729406,
        0.36146094,
        0.55695365,
        0.40899441,
        0.09194422,
        -0.00288493,
        0.34746347,
        -0.72713794,
        0.10300112,
        0.99149757,
        0.13769308,
        1.0337861,
        0.42087642,
        0.39651021,
        -0.61231673,
        0.44392543,
        -0.18680237,
        -0.41457531,
        0.53099271,
        -0.176059,
        0.10454044,
        -0.03501261,
        0.87023518,
        -0.34357893,
    ],
)

assert numpy.round(inf, 6) == numpy.round(0.99391333, 6)
qem.round_assert(
    grad_inf,
    [
        0.00290636,
        -0.00660732,
        0.01029193,
        -0.00503707,
        -0.00188473,
        0.00029818,
        -0.0009589,
        -0.00443722,
        0.00573339,
        -0.00717827,
        -0.00533586,
        0.00209591,
        -0.00158446,
        -0.00212844,
        -0.00598255,
        -0.01039361,
        0.00146131,
        -0.00012774,
        0.00447131,
        0.00702376,
        0.0047238,
        -0.00742065,
        -0.01433074,
        -0.00862076,
    ],
)


# Kagome 2x2 opl
par_multiplicity = 6
parameters = ch.Variable(
    xp.array(
        [
            6.097430185765912,
            0.5166558411077276,
            -1.5384999770341423,
            -1.9514345193067513,
            -1.9078261812912594,
            -2.7691045498263342,
        ]
    )
)

with open("data/kagome/periodic/own_graph/ops/energy/2x2/graph_input.txt", "r") as file:
    exec(file.read())

complete_graph = complete_graph_input
init_layer = init_layer_input
layers = layers_input
del complete_graph_input
del init_layer_input
del layers_input

nodes = [node for edge in complete_graph for node in edge]
nodes = set(nodes)
n = len(nodes)
del nodes

gs_reg = qem.Reg(n)
with open("data/kagome/periodic/own_graph/ops/energy/2x2/gs.dat", "rb") as file:
    gs_reg.psi.re = xp.array(pickle.load(file)).reshape((2,) * n)

init_reg = qem.Reg(n)
for edge in init_layer:
    qem.apply_prepare_singlet(edge, init_reg)

E = _HVQE.Heisenberg_energy_from_parameters(
    complete_graph, init_reg, layers, n, par_multiplicity, parameters
)
E.backward()
grad_E = parameters.grad
E = float(E.array)

parameters = ch.Variable(
    xp.array(
        [
            6.097430185765912,
            0.5166558411077276,
            -1.5384999770341423,
            -1.9514345193067513,
            -1.9078261812912594,
            -2.7691045498263342,
        ]
    )
)

inf = _HVQE.infidelity_from_parameters(
    init_reg, layers, n, par_multiplicity, parameters, gs_reg
)
inf.backward()
grad_inf = parameters.grad
inf = inf.array

assert numpy.round(E, 6) == numpy.round(-2.34581086, 6)
qem.round_assert(
    grad_E, [-0.41446951, 0.079979, 0.81810719, 0.20887126, 0.80076491, -0.28956556]
)

assert numpy.round(inf, 6) == numpy.round(0.99079512, 6)
qem.round_assert(
    grad_inf,
    [-0.01020546, -0.00552987, -0.00329207, 0.02484425, -0.01589784, 0.01016548],
)


def test_noisy_Heisenberg_energy_and_infidelity_from_parameters():
    complete_graph = [(0, 1)]
    init_reg = qem.Reg(2)
    qem.apply_H((0), init_reg)
    qem.apply_CNOT((0, 1), init_reg)
    layers = [[(0, 1)]]
    n = 2
    par_multiplicity = 1
    parameters = ch.Variable(numpy.array([0.3423]))
    pe = 0
    gs_reg = qem.Reg(2)
    qem.apply_H((0,), gs_reg)
    qem.apply_CNOT((0, 1), gs_reg)
    noise_type = "depol"
    E, inf = _HVQE.noisy_Heisenberg_energy_and_infidelity_from_parameters(
        complete_graph,
        init_reg,
        layers,
        n,
        par_multiplicity,
        parameters,
        pe,
        noise_type,
        gs_reg,
    )
    assert [E, inf] == ["no_noise", "no_noise"]

    complete_graph = [(0, 1)]
    init_reg = qem.Reg(2)
    qem.apply_H((1,), init_reg)
    qem.apply_CNOT((0, 1), init_reg)
    layers = [[(0, 1)]]
    n = 2
    par_multiplicity = 1
    parameters = ch.Variable(numpy.array([0.3423]))
    pe = 1
    noise_type = "depol"
    gs_reg = qem.Reg(2)
    qem.apply_H((0,), gs_reg)
    qem.apply_CNOT((0, 1), gs_reg)
    lst = []
    for s in range(100):
        E, inf = _HVQE.noisy_Heisenberg_energy_and_infidelity_from_parameters(
            complete_graph,
            init_reg,
            layers,
            n,
            par_multiplicity,
            parameters,
            pe,
            noise_type,
            gs_reg,
        )
        lst.append([E.data, inf.data])

    lst = numpy.array(lst).transpose()
    avE = numpy.mean(lst[0])
    avinf = numpy.mean(lst[1])

    assert numpy.around(avinf, 2) == numpy.around(1 - 1 / 2**2, 2)
    assert numpy.around(avE, 2) == numpy.around(0, 2)

    # Check that the expected number of errors is correct
    complete_graph = [(0, 1), (1, 2), (2, 0)]
    init_reg = qem.Reg(3)
    qem.apply_H((1,), init_reg)
    qem.apply_CNOT((0, 1), init_reg)
    layers = [[(1, 2)], [(2, 0)], [(0, 1)]]
    n = 3
    noise_type = "depol"
    par_multiplicity = 1
    parameters = ch.Variable(numpy.random.rand(3))
    pe = 0.1
    gs_reg = qem.Reg(3)
    qem.apply_H((0,), gs_reg)
    qem.apply_CNOT((0, 1), gs_reg)
    lst = []
    ns = 1000
    for s in range(ns):
        E, inf = _HVQE.noisy_Heisenberg_energy_and_infidelity_from_parameters(
            complete_graph,
            init_reg,
            layers,
            n,
            par_multiplicity,
            parameters,
            pe,
            noise_type,
            gs_reg,
        )
        lst.append([str(E), str(inf)])

    lst = numpy.array(lst).transpose()
    a = numpy.count_nonzero(lst[0] == "no_noise") / ns
    b = (1 - pe) ** (3 * 4)
    assert numpy.around(a, 1) == numpy.around(b, 1)

    # Check that the expected number of errors is correct, for bitflip errors.
    noise_type = "bitflip"
    complete_graph = [(0, 1), (1, 2), (2, 0)]
    init_reg = qem.Reg(3)
    qem.apply_H((1,), init_reg)
    qem.apply_CNOT((0, 1), init_reg)
    layers = [[(1, 2)], [(2, 0)], [(0, 1)]]
    n = 3
    par_multiplicity = 1
    parameters = ch.Variable(numpy.random.rand(3))
    pe = 0.2
    gs_reg = qem.Reg(3)
    qem.apply_H((0,), gs_reg)
    qem.apply_CNOT((0, 1), gs_reg)
    lst = []
    ns = 1000
    for s in range(ns):
        E, inf = _HVQE.noisy_Heisenberg_energy_and_infidelity_from_parameters(
            complete_graph,
            init_reg,
            layers,
            n,
            par_multiplicity,
            parameters,
            pe,
            noise_type,
            gs_reg,
        )
        lst.append([str(E), str(inf)])

    lst = numpy.array(lst).transpose()
    a = numpy.count_nonzero(lst[0] == "no_noise") / ns
    b = (1 - pe) ** (3 * 4)
    assert numpy.around(a, 1) == numpy.around(b, 1)


test_noisy_Heisenberg_energy_and_infidelity_from_parameters()

print("all tests passed successfully")
