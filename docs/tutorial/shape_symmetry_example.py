"""Model an infinite nanoribbon consisting of graphene rings"""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi

pb.pltutils.use_style()


def ring(inner_radius, outer_radius):
    """A simple ring shape"""
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)

    return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])


model = pb.Model(
    graphene.monolayer_4atom(),
    ring(inner_radius=1.4, outer_radius=2),  # length in nanometers
    pb.translational_symmetry(a1=3.8, a2=False)  # period in nanometers
)

plt.figure(figsize=pb.pltutils.cm2inch(20, 7))
model.plot()
plt.show()


# only solve for the 10 lowest energy eigenvalues
solver = pb.solver.arpack(model, k=10)
a = 3.8  # [nm] unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot(point_labels=[r'$-\pi / a$', r'$\pi / a$'])
plt.show()


solver.set_wave_vector(k=0)
ldos = solver.calc_spatial_ldos(energy=0, broadening=0.01)  # LDOS around 0 eV

plt.figure(figsize=pb.pltutils.cm2inch(20, 7))
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")
plt.show()


solver.set_wave_vector(k=pi/a)
ldos = solver.calc_spatial_ldos(energy=0, broadening=0.01)  # LDOS around 0 eV

plt.figure(figsize=pb.pltutils.cm2inch(20, 7))
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")
plt.show()
