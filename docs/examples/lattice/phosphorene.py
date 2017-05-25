"""Create and plot a phosphorene lattice, its Brillouin zone and band structure"""
import pybinding as pb
import matplotlib.pyplot as plt
from math import pi, sin, cos

pb.pltutils.use_style()


def phosphorene_4band():
    """Monolayer phosphorene lattice using the four-band model"""
    a = 0.222
    ax = 0.438
    ay = 0.332
    theta = 96.79 * (pi / 180)
    phi = 103.69 * (pi / 180)

    lat = pb.Lattice(a1=[ax, 0], a2=[0, ay])

    h = a * sin(phi - pi / 2)
    s = 0.5 * ax - a * cos(theta / 2)
    lat.add_sublattices(
        ('A', [-s/2,        -ay/2, h], 0),
        ('B', [ s/2,        -ay/2, 0], 0),
        ('C', [-s/2 + ax/2,     0, 0], 0),
        ('D', [ s/2 + ax/2,     0, h], 0)
    )

    lat.register_hopping_energies({
        't1': -1.22,
        't2': 3.665,
        't3': -0.205,
        't4': -0.105,
        't5': -0.055
    })

    lat.add_hoppings(
        # t1
        ([-1,  0], 'A', 'D', 't1'),
        ([-1, -1], 'A', 'D', 't1'),
        ([ 0,  0], 'B', 'C', 't1'),
        ([ 0, -1], 'B', 'C', 't1'),
        # t2
        ([ 0,  0], 'A', 'B', 't2'),
        ([ 0,  0], 'C', 'D', 't2'),
        # t3
        ([ 0,  0], 'A', 'D', 't3'),
        ([ 0, -1], 'A', 'D', 't3'),
        ([ 1,  1], 'C', 'B', 't3'),
        ([ 1,  0], 'C', 'B', 't3'),
        # t4
        ([ 0,  0], 'A', 'C', 't4'),
        ([ 0, -1], 'A', 'C', 't4'),
        ([-1,  0], 'A', 'C', 't4'),
        ([-1, -1], 'A', 'C', 't4'),
        ([ 0,  0], 'B', 'D', 't4'),
        ([ 0, -1], 'B', 'D', 't4'),
        ([-1,  0], 'B', 'D', 't4'),
        ([-1, -1], 'B', 'D', 't4'),
        # t5
        ([-1,  0], 'A', 'B', 't5'),
        ([-1,  0], 'C', 'D', 't5')
    )

    return lat

plt.figure(figsize=(6, 6))
lattice = phosphorene_4band()
lattice.plot()
plt.show()


lattice.plot_brillouin_zone()
plt.show()


model = pb.Model(phosphorene_4band(), pb.translational_symmetry())
solver = pb.solver.lapack(model)

ax = 0.438
ay = 0.332
kx = pi / ax
ky = pi / ay
bands = solver.calc_bands([kx, ky], [kx, 0], [0, 0], [0, ky], [kx, ky])
bands.plot(point_labels=["S", "Y", r"$\Gamma$", "X", "S"])
plt.show()


model.lattice.plot_brillouin_zone(decorate=False)
bands.plot_kpath(point_labels=["S", "Y", r"$\Gamma$", "X", "S"])
plt.show()
