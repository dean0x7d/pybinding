"""PN junction and broken sublattice symmetry in a graphene nanoribbon"""
import pybinding as pb
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt

pb.pltutils.use_style()


def mass_term(delta):
    """Break sublattice symmetry with opposite A and B onsite energy"""
    @pb.onsite_energy_modifier
    def potential(energy, sub_id):
        energy[sub_id == 'A'] += delta
        energy[sub_id == 'B'] -= delta
        return energy

    return potential


def pn_juction(y0, v1, v2):
    """PN junction potential

    The `y0` argument is the position of the junction, while `v1` and `v2`
    are the values of the potential (in eV) before and after the junction.
    """
    @pb.onsite_energy_modifier
    def potential(energy, y):
        energy[y < y0] += v1
        energy[y >= y0] += v2
        return energy

    return potential


model = pb.Model(
    graphene.monolayer(),
    pb.rectangle(1.2),  # width in nanometers
    pb.translational_symmetry(a1=True, a2=False),
    mass_term(delta=2.5),  # eV
    pn_juction(y0=0, v1=-2.5, v2=2.5)  # y0 in [nm] and v1, v2 in [eV]
)
model.plot()
plt.show()


# plot the potential: note that pn_junction cancels out delta on some sites
model.onsite_map.plot(cmap="coolwarm", site_radius=0.04)
pb.pltutils.colorbar(label="U (eV)")
plt.show()

# compute the bands
solver = pb.solver.lapack(model)
a = graphene.a_cc * sqrt(3)  # nanoribbon unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot()
plt.show()
