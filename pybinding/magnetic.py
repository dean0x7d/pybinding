import numpy as _np
from pybinding import modifier


def constant(magnetic_field):
    from pybinding.constant import pi, phi0
    scale = 1e-18  # both the vector potential and coordinates are in [nm] -> scale to [m]
    const = scale * 2 * pi / phi0

    @modifier.hopping_energy
    def hop(hopping, x1, y1, z1, x2, y2, z2):
        # vector potential
        ax = magnetic_field * (y1 + y2) / 2
        # integral of (A * dl) from positon 1 to position 2
        peierls = ax * (x1 - x2)
        return hopping * _np.exp(1j * const * peierls)

    return hop
