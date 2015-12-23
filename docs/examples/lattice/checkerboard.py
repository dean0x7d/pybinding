"""Two dimensional checkerboard lattice with real hoppings"""
import pybinding as pb
import matplotlib.pyplot as plt
from math import pi

pb.pltutils.use_style()


def checkerboard(d=0.2, delta=1.1, t=0.6):
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_sublattices(
        ('A', [0, 0], -delta),
        ('B', [d/2, d/2], delta)
    )
    lat.add_hoppings(
        ([ 0,  0], 'A', 'B', t),
        ([ 0, -1], 'A', 'B', t),
        ([-1,  0], 'A', 'B', t),
        ([-1, -1], 'A', 'B', t)
    )
    return lat

lattice = checkerboard()
lattice.plot()
plt.show()

lattice.plot_brillouin_zone()
plt.show()

model = pb.Model(checkerboard(), pb.translational_symmetry())
solver = pb.solver.lapack(model)

bands = solver.calc_bands([0, 0], [0, 5*pi], [5*pi, 5*pi], [0, 0])
bands.plot()
plt.show()
