import math
import pybinding as pb
from .constants import a, a_cc

__all__ = ['hexagon_ac']


def hexagon_ac(side_width, offset=(-a/2, 0)):
    """Hexagon aligned with the armchair edges of graphene

    Parameters
    ----------
    side_width : float
        Hexagon side width. It will be adjusted to make perfect armchair edges.
    offset : array_like
        The default value makes sure that a carbon hexagon is at the center
        if the :func:`.monolayer()` lattice is used.
    """
    # expected number of atoms on an armchair edge
    side_atoms = math.ceil((side_width / a_cc + 1) * 2/3)
    side_atoms += (side_atoms % 2)  # it must be an even number
    # set a better width value based on the calculated number of atoms
    side_width = (3/2 * side_atoms - 1) * a_cc - a_cc / 2

    x0 = side_width * math.sqrt(3) / 2
    y0 = side_width
    hexagon = pb.Polygon([(0,  y0), ( x0,  y0/2), ( x0, -y0/2),
                          (0, -y0), (-x0, -y0/2), (-x0,  y0/2)],
                         offset)
    return hexagon
