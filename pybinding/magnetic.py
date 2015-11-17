import numpy as _np
import pybinding as pb

__all__ = ['constant']


def constant(magnetic_field):
    from pybinding.constants import pi, phi0
    scale = 1e-18  # both the vector potential and coordinates are in [nm] -> scale to [m]
    const = scale * 2 * pi / phi0

    @pb.hopping_energy_modifier
    def hop(hopping, x1, y1, x2, y2):
        # vector potential
        ax = magnetic_field * (y1 + y2) / 2
        # integral of (A * dl) from position 1 to position 2
        peierls = ax * (x1 - x2)
        return hopping * _np.exp(1j * const * peierls)

    return hop
