import numpy as _np
import pybinding as pb
from .constants import a_cc, beta, hbar

__all__ = ['gaussian_bump', 'triaxial_strain']


@pb.hopping_energy_modifier
def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
    l = _np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    w = l / a_cc - 1
    return energy * _np.exp(-beta * w)


def gaussian_bump(h0, r_limit):
    import math
    b = r_limit * math.sqrt(2) / 3.5

    @pb.site_position_modifier
    def displacement(x, y):
        r2 = x**2 + y**2
        r = _np.sqrt(r2)

        z = h0 * _np.exp(-r2 / b**2)
        z *= (r < r_limit)

        return x, y, z

    return displacement, strained_hopping


def triaxial_strain(magnetic_field):
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
