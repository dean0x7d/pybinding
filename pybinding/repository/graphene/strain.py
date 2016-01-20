import numpy as np
import pybinding as pb
from .constants import a_cc, hbar

__all__ = ['gaussian_bump', 'triaxial_strain']

beta = 3.37  #: strain hopping modulation


@pb.hopping_energy_modifier
def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
    l = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    w = l / a_cc - 1
    return energy * np.exp(-beta * w)


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


def triaxial_strain(magnetic_field):
    """Triaxial strain corresponding to a homogeneous pseudo-magnetic field

    Parameters
    ----------
    magnetic_field : float
        Intensity of the pseudo-magnetic field to induce.
    """

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
