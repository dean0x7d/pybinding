import numpy as np
import pybinding as pb
from pybinding.constants import pi, phi0, hbar

__all__ = ['mass_term', 'coulomb_potential', 'constant_magnetic_field',
           'triaxial_strain', 'gaussian_bump']


def mass_term(delta):
    """Break sublattice symmetry, make massive Dirac electrons, open a band gap

    Only for monolayer graphene.

    Parameters
    ----------
    delta : float
        Onsite energy +delta is added to sublattice 'A' and -delta to 'B'.
    """
    @pb.onsite_energy_modifier
    def onsite(energy, sub_id):
        energy[sub_id == "A"] += delta
        energy[sub_id == "B"] -= delta
        return energy

    return onsite


def coulomb_potential(beta, cutoff_radius=.0, offset=(0, 0, 0)):
    """A Coulomb potential created by an impurity in graphene

    Parameters
    ----------
    beta : float
       Charge of the impurity [unitless].
    cutoff_radius : float
        Cut off the potential below this radius [nm].
    offset: array_like
        Position of the charge.
    """
    from .constants import vf
    # beta is dimensionless -> multiply hbar*vF makes it [eV * nm]
    scaled_beta = beta * hbar * vf

    @pb.onsite_energy_modifier
    def potential(energy, x, y, z):
        x0, y0, z0 = offset
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        r[r < cutoff_radius] = cutoff_radius
        return energy - scaled_beta / r

    return potential


def constant_magnetic_field(magnitude):
    """Constant magnetic field in the z-direction, perpendicular to the graphene plane

    Parameters
    ----------
    magnitude : float
        In units of Tesla.
    """
    scale = 1e-18  # both the vector potential and coordinates are in [nm] -> scale to [m]
    const = scale * 2 * pi / phi0

    @pb.hopping_energy_modifier(is_complex=True)
    def func(energy, x1, y1, x2, y2):
        # vector potential along the x-axis
        vp_x = 0.5 * magnitude * (y1 + y2)
        # integral of (A * dl) from position 1 to position 2
        peierls = vp_x * (x1 - x2)
        return energy * np.exp(1j * const * peierls)

    return func


@pb.hopping_energy_modifier
def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
    from .constants import a_cc, beta

    l = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    w = l / a_cc - 1
    return energy * np.exp(-beta * w)


def triaxial_strain(magnetic_field):
    """Triaxial strain corresponding to a homogeneous pseudo-magnetic field

    Parameters
    ----------
    magnetic_field : float
        Intensity of the pseudo-magnetic field to induce.
    """
    from .constants import a_cc, beta

    def field_to_strain(field):
        return field * a_cc / (4 * hbar * beta) * 10**-18

    c = field_to_strain(magnetic_field)
    zigzag_direction = False

    @pb.site_position_modifier
    def displacement(x, y, z):
        if not zigzag_direction:
            ux = 2*c * x*y
            uy = c * (x**2 - y**2)
        else:
            ux = c * (y**2 - x**2)
            uy = 2*c * x*y

        x += ux
        y += uy

        return x, y, z

    return displacement, strained_hopping


def gaussian_bump(height, sigma, center=(0, 0)):
    """Gaussian bump deformation

    Parameters
    ----------
    height : float
        Height of the bump [nm].
    sigma : float
        Gaussian sigma parameter: controls the width of the bump [nm].
    center : array_like
        Position of the center of the bump.
    """
    @pb.site_position_modifier
    def displacement(x, y):
        x0, y0 = center
        r2 = (x - x0)**2 + (y - y0)**2
        z = height * np.exp(-r2 / sigma**2)
        return x, y, z

    return displacement, strained_hopping
