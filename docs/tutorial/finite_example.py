"""Model a graphene ring structure and calculate the local density of states"""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene

pb.pltutils.use_style()


def ring(inner_radius, outer_radius):
    """A simple ring shape"""
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)

    return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])


model = pb.Model(
    graphene.monolayer(),
    ring(inner_radius=1.4, outer_radius=2)  # length in nanometers
)

model.plot()
plt.show()


# only solve for the 20 lowest energy eigenvalues
solver = pb.solver.arpack(model, k=20)
ldos = solver.calc_spatial_ldos(energy=0, broadening=0.05)  # LDOS around 0 eV
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")
plt.show()
