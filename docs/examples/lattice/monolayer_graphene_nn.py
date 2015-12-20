"""Monolayer graphene with next-nearest hoppings"""
import pybinding as pb
import matplotlib.pyplot as plt
from math import sqrt

pb.pltutils.use_style()


def monolayer_graphene_nn():
    a = 0.24595   # (nm) unit cell length
    a_cc = 0.142  # (nm) carbon-carbon distance
    t = -2.8      # (eV) nearest neighbour hopping
    t_nn = 0.1    # (eV) next-nearest neighbour hopping

    lat = pb.Lattice(
        a1=[a, 0],
        a2=[a/2, a/2 * sqrt(3)]
    )
    lat.add_sublattices(
        ('A', [0, -a_cc/2]),
        ('B', [0,  a_cc/2])
    )
    lat.add_hoppings(
        # between A and B inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t),
        # next-nearest
        ([1,  0], 'A', 'A', t_nn),
        ([1,  0], 'B', 'B', t_nn),
        ([0,  1], 'A', 'A', t_nn),
        ([0,  1], 'B', 'B', t_nn),
        ([1, -1], 'A', 'A', t_nn),
        ([1, -1], 'B', 'B', t_nn)
    )
    return lat

lattice = monolayer_graphene_nn()
lattice.plot()
plt.show()

lattice.plot_brillouin_zone()
plt.show()
