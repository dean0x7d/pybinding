import numpy as np
import pybinding as pb

__all__ = ['coulomb_potential']


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
    from .constants import hbar, vf
    # beta is dimensionless -> multiply hbar*vF makes it [eV * nm]
    scaled_beta = beta * hbar * vf

    @pb.onsite_energy_modifier
    def potential(energy, x, y, z):
        x0, y0, z0 = offset
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        r[r < cutoff_radius] = cutoff_radius
        return energy - scaled_beta / r

    return potential
