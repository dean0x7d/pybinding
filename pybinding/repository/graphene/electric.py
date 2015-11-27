import numpy as _np
import pybinding as pb


def coulomb(beta, r_const=.0, offset=(0, 0, 0)):
    """A Coulomb potential created by an impurity in graphene.


    Parameters
    ----------
    beta : float
       Charge of the impurity [unitless].
    r_const : float, optional
        Radius inside which the potential should be constant [nm].
    offset: tuple of floats, optional
        Position of the charge.
    """
    from .constants import hbar, vf
    # beta is dimensionless -> multiply hbar*vF makes it [eV * nm]
    scaled_beta = beta * hbar * vf

    @pb.onsite_energy_modifier
    def potential(energy, x, y, z):
        x0, y0, z0 = offset
        r = _np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        r[r < r_const] = r_const
        return energy - scaled_beta / r

    return potential
