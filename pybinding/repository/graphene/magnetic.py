import numpy as np
import pybinding as pb
from pybinding.constants import pi, phi0

__all__ = ['constant_magnetic_field']


def constant_magnetic_field(magnitude):
    """Constant magnetic field in the z-direction, perpendicular to the graphene plane

    Parameters
    ----------
    magnitude : float
        In units of Tesla.
    """
    scale = 1e-18  # both the vector potential and coordinates are in [nm] -> scale to [m]
    const = scale * 2 * pi / phi0

    @pb.hopping_energy_modifier
    def function(energy, x1, y1, x2, y2):
        # vector potential along the x-axis
        vp_x = 0.5 * magnitude * (y1 + y2)
        # integral of (A * dl) from position 1 to position 2
        peierls = vp_x * (x1 - x2)
        return energy * np.exp(1j * const * peierls)

    return function
