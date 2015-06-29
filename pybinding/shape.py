import numpy as np
import matplotlib.pyplot as plt

import _pybinding
from . import pltutils
from .utils import with_defaults

__all__ = ['Polygon', 'Circle', 'primitive', 'rectangle', 'regular_polygon', 'circle']


class Polygon(_pybinding.Polygon):
    """Shape defined by a list of vertices

    Parameters
    ----------
    *vertices : tuple
        Polygon vertices. Must be defined in clockwise or counterclockwise order.
    """

    def __init__(self, *vertices):
        super().__init__()
        if not vertices:
            return

        # vertices may be specified directly as arguments
        # or as an iterable in the first argument
        self.vertices = vertices if len(vertices) > 1 else vertices[0]

    @property
    def vertices(self):
        return zip(self.x, self.y)

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


class Circle(_pybinding.Circle):
    def plot(self, **kwargs):
        plt.gca().add_artist(plt.Circle(tuple(self.center), self.r, fill=False,
                                        **with_defaults(kwargs, color='black')))
        plt.axis('scaled')
        pltutils.despine(trim=True)
        pltutils.add_margin()


def primitive(v1=None, v2=None, v3=None, nanometers=False):
    """Shape of the lattice's primitive unit cell.

    Parameters
    ----------
    v1, v2, v3 : int or float
        Number of unit vector lengths in the respective primitive vector directions.

    nanometers : bool
        If set to True, take length in nanometers instead of number of unit vector lengths.
    """

    lengths = tuple((v or 0) for v in (v1, v2, v3))
    return _pybinding.Primitive(lengths, nanometers)


def rectangle(x, y=None):
    y = y if y else x
    x0 = x / 2
    y0 = y / 2
    return Polygon([x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0])


def regular_polygon(num_sides, radius, angle=0):
    from math import pi, sin, cos
    angles = [angle + 2 * n * pi / num_sides for n in range(num_sides)]
    return Polygon((radius * sin(a), radius * cos(a)) for a in angles)


def circle(radius, center=(0, 0, 0)):
    return Circle(radius, center)
