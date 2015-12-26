import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import with_defaults

__all__ = ['Polygon', 'primitive', 'rectangle', 'regular_polygon', 'circle',
           'translational_symmetry']


class Polygon(_cpp.Polygon):
    """Shape defined by a list of vertices

    Parameters
    ----------
    vertices : list of array_like
        Polygon vertices. Must be defined in clockwise or counterclockwise order.
    """

    def __init__(self, vertices):
        super().__init__()
        self.vertices = vertices

    @property
    def vertices(self):
        return list(zip(self.x, self.y))

    @vertices.setter
    def vertices(self, vertices):
        x, y = zip(*vertices)
        if len(x) < 3:
            raise ValueError("A polygon must have at least 3 sides")

        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def plot(self, **kwargs):
        plt.plot(np.append(self.x, self.x[0]), np.append(self.y, self.y[0]),
                 **with_defaults(kwargs, color='black'))
        plt.axis('scaled')
        pltutils.despine(trim=True)
        pltutils.add_margin()


class Circle(_cpp.Circle):
    def plot(self, **kwargs):
        plt.gca().add_artist(plt.Circle(tuple(self.center), self.r, fill=False,
                                        **with_defaults(kwargs, color='black')))
        plt.axis('scaled')
        pltutils.despine(trim=True)
        pltutils.add_margin()


def primitive(a1=1, a2=1, a3=1):
    """Repeat the lattice unit cell a number of times

    Parameters
    ----------
    a1, a2, a3 : int or float
        Number of times to repeat the unit cell in the respective lattice vector directions.
    """
    return _cpp.Primitive(a1, a2, a3)


def rectangle(x, y=None):
    """A simple rectangle shape

    Parameters
    ----------
    x : float
        Width of the rectangle.
    y : float, optional
        Height of the rectangle. If not give, assumed equal to `x`.
    """
    y = y or x
    x0, y0 = x / 2, y / 2
    return Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])


def regular_polygon(num_sides, radius, angle=0):
    """Regular polygon

    Parameters
    ----------
    num_sides : int
        Number of sides.
    radius : float
        Radius of the circle which connects all the vertices of the polygon.
    angle : float
        Rotate the polygon.
    """
    from math import pi, sin, cos
    angles = [angle + 2 * n * pi / num_sides for n in range(num_sides)]
    return Polygon([(radius * sin(a), radius * cos(a)) for a in angles])


def circle(radius, center=(0, 0, 0)):
    """Perfect circle

    Parameters
    ----------
    radius : float
    center : array_like
        Position of the center.
    """
    return Circle(radius, center)


def translational_symmetry(a1=True, a2=True, a3=True):
    """Simple translational symmetry

    Parameters
    ----------
    a1, a2, a3 : bool or float
        Control translation in the 'a1, a2, a3' lattice vector directions.
        Possible values:

        * False -> No translational symmetry in this direction.
        * True -> Translation length is automatically set to the unit cell length.
        * float value -> Manually set the translation length in nanometers.
    """
    def to_cpp_params(value):
        if value is False:
            return -1  # disabled
        elif value is True:
            return 0  # automatic length
        else:
            return value  # manual length

    lengths = tuple(to_cpp_params(a) for a in (a1, a2, a3))
    return _cpp.Translational(lengths)
