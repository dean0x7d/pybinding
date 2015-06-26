from pybinding.shape import Polygon
import math
from .constants import a, a_cc


def hexagon_ac(side_width, offset=(-a/2, 0)):
    """ Hexagon aligned with the armchair edges of graphene """
    hexagon = Polygon()

    # -> the side width needs to be adjusted to make a perfect hexagon with armchair edges
    # calculate number of atoms on armchair edge for the given width
    side_atoms = math.ceil((side_width / a_cc + 1) * 2/3)
    side_atoms += (side_atoms % 2)  # round to an even number
    # set better width value based on the calculated number of atoms
    side_width = (3/2 * side_atoms - 1) * a_cc - a_cc / 2

    x0 = side_width * math.sqrt(3) / 2
    y0 = side_width

    hexagon.vertices = (0, y0), (x0, y0/2), (x0, -y0/2), (0, -y0), (-x0, -y0/2), (-x0, y0/2)

    # make sure that: shape center == carbon hexagon center
    hexagon.offset = offset
    return hexagon
