import math
import pybinding as pb
from .constants import a, a_cc

__all__ = ['hexagon_ac']


def hexagon_ac(side_width, lattice_offset=(-a/2, 0)):
    """A graphene-specific shape which guaranties armchair edges on all sides

    Parameters
    ----------
    side_width : float
        Hexagon side width. It may be adjusted slightly to ensure armchair edges.
    lattice_offset : array_like
        Offset the lattice so a carbon hexagon is at the center of the shape.
        The default value is specific for :func:`.monolayer()` and :func:`.bilayer()`
        lattices from this material repository.
    """
    # expected number of atoms on an armchair edge
    side_atoms = math.ceil((side_width / a_cc + 1) * 2/3)
    side_atoms += (side_atoms % 2)  # it must be an even number
    # set a better width value based on the calculated number of atoms
    side_width = (3/2 * side_atoms - 1) * a_cc - a_cc / 2

    x0 = side_width * math.sqrt(3) / 2
    y0 = side_width
    hexagon = pb.Polygon([(0,  y0), ( x0,  y0/2), ( x0, -y0/2),
                          (0, -y0), (-x0, -y0/2), (-x0,  y0/2)])
    hexagon.lattice_offset = lattice_offset
    return hexagon
