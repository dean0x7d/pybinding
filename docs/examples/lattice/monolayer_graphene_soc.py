"""Calculate the band structure of graphene with Rashba spin-orbit coupling"""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt


def monolayer_graphene_soc():
    """Return the lattice specification for monolayer graphene with Rashba SOC,
       see http://doi.org/10.1103/PhysRevB.95.165415 for reference"""
    from pybinding.constants import pauli
    from pybinding.repository.graphene import a_cc, a, t

    onsite = 0.05  # [eV] onsite energy
    rashba = 0.1   # [eV] strength of Rashba SOC
    rashba_so = 1j * 2/3 * rashba

    # create a lattice with 2 primitive vectors
    a1 = np.array([a / 2 * sqrt(3), a / 2])
    a2 = np.array([a / 2 * sqrt(3), -a / 2])
    lat = pb.Lattice(
        a1=a1, a2=a2
    )

    pos_a = np.array([-a_cc / 2, 0])
    pos_b = np.array([+a_cc / 2, 0])

    lat.add_sublattices(
        ('A', pos_a, [[ onsite, 0], [0,  onsite]]),
        ('B', pos_b, [[-onsite, 0], [0, -onsite]]))

    # nearest neighbor vectors
    d1 = (pos_b - pos_a) / a_cc          # [ 0,  0]
    d2 = (pos_b - pos_a - a1) / a_cc     # [-1,  0]
    d3 = (pos_b - pos_a - a2) / a_cc     # [ 0, -1]

    nn_hopp = np.array([[t, 0], [0, t]])                            # nn hopping, same spin
    t1 = nn_hopp + rashba_so * (pauli.x * d1[1] - pauli.y * d1[0])  # cross([sx , sy], [dx, dy])
    t2 = nn_hopp + rashba_so * (pauli.x * d2[1] - pauli.y * d2[0])
    t3 = nn_hopp + rashba_so * (pauli.x * d3[1] - pauli.y * d3[0])

    # name and position
    lat.add_hoppings(
        ([0,  0], 'A', 'B', t1),
        ([-1, 0], 'A', 'B', t2),
        ([0, -1], 'A', 'B', t3)
    )

    return lat


lattice = monolayer_graphene_soc()
lattice.plot()
plt.show()

lattice.plot_brillouin_zone()
plt.show()

model = pb.Model(lattice, pb.translational_symmetry())
solver = pb.solver.lapack(model)

k_points = model.lattice.brillouin_zone()
Gamma = [0, 0]
K1 = k_points[0]
K2 = k_points[2]
M = (k_points[0] + k_points[1]) / 2

bands = solver.calc_bands(K1, Gamma, M, K2)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
plt.show()
