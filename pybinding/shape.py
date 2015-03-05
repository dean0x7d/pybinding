import _pybinding
import numpy as _np
import matplotlib.pyplot as _plt


class Polygon(_pybinding.Polygon):
    @property
    def vertices(self):
        return zip(self.x, self.y)

    @vertices.setter
    def vertices(self, vertices):
        x, y = zip(*vertices)

        self.x = _np.array(x, dtype=_np.float32)
        self.y = _np.array(y, dtype=_np.float32)

    def plot(self, **kwargs):
        defaults = dict(color='black')
        kwargs = dict(defaults, **kwargs)
        _plt.plot(_np.append(self.x, self.x[0]), _np.append(self.y, self.y[0]), **kwargs)


class Circle(_pybinding.Circle):
    def plot(self, **kwargs):
        defaults = dict(color='black')
        kwargs = dict(defaults, **kwargs)
        _plt.gca().add_artist(_plt.Circle(tuple(self.center), self.r, fill=False, **kwargs))


def primitive(v1=None, v2=None, v3=None, nanometers=False):
    """
    Shape of the lattice's primitive unit cell.

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
    y = y if y is not None else x
    x0 = x / 2
    y0 = y / 2

    rect = Polygon()
    rect.vertices = (x0, y0), (x0, -y0), (-x0, -y0), (-x0, y0)
    return rect


def hexagon(side_width):
    import math
    x0 = side_width * math.sqrt(3)/2
    y0 = side_width

    h = Polygon()
    h.vertices = (0, y0), (x0, y0/2), (x0, -y0/2), (0, -y0), (-x0, -y0/2), (-x0, y0/2)
    h.offset = (0, 0)
    return h


def circle(radius, center=(0, 0, 0)):
    return Circle(radius, center)
