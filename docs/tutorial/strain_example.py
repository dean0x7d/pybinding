"""Strain a triangular system by pulling on its vertices"""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi

pb.pltutils.use_style()


def triaxial_strain(c):
    """Strain-induced displacement and hopping energy modification"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 2*c * x*y
        uy = c * (x**2 - y**2)
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-3.37 * w)

    return displacement, strained_hopping


model = pb.Model(
    graphene.monolayer(),
    pb.regular_polygon(num_sides=3, radius=2, angle=pi),
    triaxial_strain(c=0.1)
)
model.plot()
plt.show()


plt.figure(figsize=(7, 2.5))
grid = plt.GridSpec(nrows=1, ncols=2)
for block, energy in zip(grid, [0, 0.25]):
    plt.subplot(block)
    plt.title("E = {} eV".format(energy))

    solver = pb.solver.arpack(model, k=30, sigma=energy)
    ldos_map = solver.calc_spatial_ldos(energy=energy, broadening=0.03)
    ldos_map.plot()
    pb.pltutils.colorbar(label="LDOS")

plt.show()
