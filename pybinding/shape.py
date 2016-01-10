import itertools

import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import with_defaults

__all__ = ['Polygon', 'FreeformShape', 'primitive', 'rectangle', 'regular_polygon', 'circle',
           'translational_symmetry']


class Polygon(_cpp.Polygon):
    """Shape defined by a list of vertices

    Parameters
    ----------
    vertices : list of array_like
        Polygon vertices. Must be defined in clockwise or counterclockwise order.
    offset : array_like
        Offset of the lattice coordinate origin with relative to the shape origin.
    """
    def __init__(self, vertices, offset=(0, 0, 0)):
        if len(vertices) < 3:
            raise RuntimeError("A polygon must have at least 3 sides")
        super().__init__(vertices, offset)

    @property
    def vertices(self):
        return [(x, y) for x, y, _ in super().vertices]

    def plot(self, **kwargs):
        """Line plot of the polygon

        Parameters
        ----------
        **kwargs
            Forwarded to `plt.plot()`.
        """
        x, y = zip(*self.vertices)
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), **with_defaults(kwargs, color='black'))
        plt.axis('scaled')
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        pltutils.despine(trim=True)
        pltutils.add_margin()


class FreeformShape(_cpp.Shape):
    """Shape defined by a bounding box and a function

    Parameters
    ----------
    contains : callable
        The function which selects if a point is contained within the shape.
    width : array_like
        Width up to 3 dimensions which specifies the size of the bounding box.
    center : array_like
        The position of the center of the bounding box.
    offset : array_like
        Offset of the lattice coordinate origin with relative to the shape origin.
    """
    def __init__(self, contains, width, center=(0, 0, 0), offset=(0, 0, 0)):
        width = np.atleast_1d(width)
        width.resize(3)
        center = np.atleast_1d(center)
        center.resize(3)
        # e.g. vertex == [x0, y0]
        vertex = center + width / 2
        # e.g. vertices == [(x0, y0), (x0, -y0), (-x0, y0), (-x0, -y0)]
        vertices = list(itertools.product(*zip(vertex, -vertex)))

        super().__init__(vertices, contains, np.atleast_1d(offset))
        self.contains = contains

    def plot(self, resolution=(1000, 1000), **kwargs):
        """Plot an lightly shaded silhouette of the freeform shape

        This method only works for 2D shapes.

        Parameters
        ----------
        resolution : Tuple[int, int]
            The (x, y) pixel resolution of the generated shape image.
        **kwargs
            Forwarded to `plt.imshow()`.
        """
        if any(z != 0 for _, _, z in self.vertices):
            raise RuntimeError("This method only works for 2D shapes.")

        x, y, *_ = zip(*self.vertices)
        xx, yy = np.meshgrid(np.linspace(min(x), max(x), resolution[0]),
                             np.linspace(min(y), max(y), resolution[1]))
        img = self.contains(xx, yy, 0)
        img = np.ma.masked_array(img, np.logical_not(img))

        plt.imshow(img, extent=(min(x), max(x), min(y), max(y)),
                   **with_defaults(kwargs, cmap='gray', alpha=0.15))
        plt.axis('scaled')
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
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
    def contains(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2) < radius

    return FreeformShape(contains, [2 * radius, 2 * radius], center)


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
    return _cpp.TranslationalSymmetry(lengths)
