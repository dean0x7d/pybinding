"""System shape and symmetry"""
import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import with_defaults

__all__ = ['FreeformShape', 'Polygon', 'CompositeShape',
           'circle', 'line', 'primitive', 'rectangle', 'regular_polygon',
           'translational_symmetry']


def _plot_freeform_shape(vertices, contains, resolution=(1000, 1000), **kwargs):
    """Plot the area where `contains(x, y, z)` is True within the polygon given by `vertices`

    Parameters
    ----------
    resolution : Tuple[int, int]
        The (x, y) pixel resolution of the generated image.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.imshow`.
    """
    if any(z != 0 for _, _, z in vertices):
        raise RuntimeError("This method only works for 2D shapes.")

    x, y, *_ = zip(*vertices)
    xx, yy = np.meshgrid(np.linspace(min(x), max(x), resolution[0], dtype=np.float32),
                         np.linspace(min(y), max(y), resolution[1], dtype=np.float32))
    area = contains(xx.flat, yy.flat, np.zeros_like(xx.flat))
    area.shape = xx.shape
    area = np.ma.masked_array(area, np.logical_not(area))
    area = np.flipud(area)

    if "color" in kwargs:
        kwargs["cmap"], kwargs["norm"] = pltutils.direct_cmap_norm([1], [kwargs.pop("color")])

    im = plt.imshow(area, extent=(min(x), max(x), min(y), max(y)),
                    **with_defaults(kwargs, cmap="gray", alpha=0.15, interpolation="bicubic"))

    plt.axis("scaled")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    pltutils.despine(trim=True)
    pltutils.add_margin()

    return im


class _CompositionMixin:
    """Provides logical and arithmetic operators to form composite shapes"""
    def __and__(self, other):
        return CompositeShape(self, other, np.logical_and)

    def __or__(self, other):
        return CompositeShape(self, other, np.logical_or)

    def __xor__(self, other):
        return CompositeShape(self, other, np.logical_xor)

    def __add__(self, other):
        return self.__or__(other)

    def __sub__(self, other):
        return CompositeShape(self, other, lambda a, b: np.logical_and(a, np.logical_not(b)))


class Line(_cpp.Line):
    """Shape defined by two points

    This is intended for 1D lattices or for specifying leads for 2D lattices

    Attributes
    ----------
    a, b : Union[float, array_like]
        Start and end points.
    """
    def __init__(self, a, b):
        a, b = map(np.array, (a, b))
        a.resize(2)
        b.resize(2)
        super().__init__(a, b)
        self.a = a
        self.b = b

    def with_offset(self, vector):
        """Return a copy that's offset by the given vector"""
        return Line(self.a + vector[0], self.b + vector[1])

    def plot(self, **kwargs):
        """Show the line

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        plt.plot(*zip(self.a, self.b), **with_defaults(kwargs, color='black', lw=1.6))


class Polygon(_cpp.Polygon, _CompositionMixin):
    """Shape defined by a list of vertices in a 2D plane

    Attributes
    ----------
    vertices : List[array_like]
        Must be defined in clockwise or counterclockwise order.
    """
    def __init__(self, vertices):
        if len(vertices) < 3:
            raise RuntimeError("A polygon must have at least 3 sides")
        super().__init__(vertices)

    def with_offset(self, vector):
        """Return a copy that's offset by the given vector"""
        v = np.zeros(3)
        v[:len(vector)] = vector
        return Polygon([v0 + v for v0 in self.vertices])

    def plot(self, **kwargs):
        """Line plot of the polygon

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        x, y, _ = zip(*self.vertices)
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), **with_defaults(kwargs, color='black'))
        plt.axis("scaled")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        pltutils.despine(trim=True)
        pltutils.add_margin()


class FreeformShape(_cpp.FreeformShape, _CompositionMixin):
    """Shape in 1 to 3 dimensions, defined by a function and a bounding box

    Note that this class can describe 3D shapes, but the :meth:`.plot` method can currently
    only draw in 2D. Nevertheless, a :class:`.Model` will accept 3D shapes without a problem.

    Parameters
    ----------
    contains : callable
        The function which selects if a point is contained within the shape.
    width : array_like
        Width up to 3 dimensions which specifies the size of the bounding box.
    center : array_like
        The position of the center of the bounding box.
    """
    def __init__(self, contains, width, center=(0, 0, 0)):
        super().__init__(contains, width, center)
        self.width = np.atleast_1d(width)
        self.center = np.atleast_1d(center)

    def with_offset(self, vector):
        """Return a copy that's offset by the given vector"""
        def contains(x, y, z):
            r0 = [x, y, z]
            r = [v0 - v for v0, v in zip(r0, vector)] + r0[len(vector):]
            return self.contains(*r)
        return FreeformShape(contains, self.width, self.center[:len(vector)] + vector)

    def plot(self, resolution=(1000, 1000), **kwargs):
        """Plot a lightly shaded silhouette of the freeform shape

        This method only works for 2D shapes.

        Parameters
        ----------
        resolution : Tuple[int, int]
            The (x, y) pixel resolution of the generated shape image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        return _plot_freeform_shape(self.vertices, self.contains, resolution, **kwargs)


class CompositeShape(_cpp.Shape, _CompositionMixin):
    """A composition of 2 shapes using some operator (and, or, xor...)
    
    This shape is usually not created directly but present the result of 
    applying logical or arithmetic operators on other shapes.
    
    Parameters
    ----------
    shape1, shape2 : _cpp.Shape
        The shapes which shall be composed.
    op : Callable
        A logical operator (and, or, xor...) to use for the composition. 
    """
    def __init__(self, shape1, shape2, op):
        from scipy.spatial import ConvexHull

        # The bounding vertices are always taken as the convex hull of the combined vertices.
        # This is a wasteful in some situations (e.g. intersections) but its always safe and
        # quick to code using scipy's implementation. Performance isn't an issue here for now
        # but revisit this if it becomes a concern.
        hull = ConvexHull(np.array(shape1.vertices + shape2.vertices)[:, :2])
        vertices = hull.points[hull.vertices]

        super().__init__(vertices, lambda x, y, z: op(shape1.contains(x, y, z),
                                                      shape2.contains(x, y, z)))

    def with_offset(self, vector):
        """Return a copy that's offset by the given vector"""
        def contains(x, y, z):
            r0 = [x, y, z]
            r = [v0 - v for v0, v in zip(r0, vector)] + r0[len(vector):]
            return self.contains(*r)

        shape = CompositeShape.__new__(CompositeShape)
        v = np.zeros(3)
        v[:len(vector)] = vector
        vertices = [v0 + v for v0 in self.vertices]
        super(CompositeShape, shape).__init__(vertices, contains)
        return shape

    def plot(self, resolution=(1000, 1000), **kwargs):
        """Plot a lightly shaded silhouette of the composite shape

        Parameters
        ----------
        resolution : Tuple[int, int]
            The (x, y) pixel resolution of the generated image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        return _plot_freeform_shape(self.vertices, self.contains, resolution, **kwargs)


def primitive(a1=1, a2=1, a3=1):
    """Follow the primitive lattice shape -- just repeat the unit cell a number of times

    Parameters
    ----------
    a1, a2, a3 : int or float
        Number of times to repeat the unit cell in the respective lattice vector directions.

    Returns
    -------
    :class:`~_pybinding.Primitive`
    """
    return _cpp.Primitive(a1, a2, a3)


def line(a, b):
    """A line shape intended for 1D lattices or to specify leads for 2D lattices

    Parameters
    ----------
    a, b : Union[float, array_like]
        Start and end points.

    Returns
    -------
    :class:`~pybinding.shape.Line`
    """
    return Line(a, b)


def rectangle(x, y=None):
    """A rectangle in the xy plane

    Parameters
    ----------
    x : float
        Width of the rectangle.
    y : float, optional
        Height of the rectangle. If not given, assumed equal to `x`.

    Returns
    -------
    :class:`~pybinding.Polygon`
    """
    y = y or x
    x0, y0 = x / 2, y / 2
    return Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])


def regular_polygon(num_sides, radius, angle=0):
    """A polygon shape where all sides have equal length

    Parameters
    ----------
    num_sides : int
        Number of sides.
    radius : float
        Radius of the circle which connects all the vertices of the polygon.
    angle : float
        Rotate the polygon.

    Returns
    -------
    :class:`~pybinding.Polygon`
    """
    from math import pi, sin, cos
    angles = [angle + 2 * n * pi / num_sides for n in range(num_sides)]
    return Polygon([(radius * sin(a), radius * cos(a)) for a in angles])


def circle(radius, center=(0, 0)):
    """A circle in the xy plane

    Parameters
    ----------
    radius : float
    center : array_like

    Returns
    -------
    :class:`~pybinding.FreeformShape`
    """
    def contains(x, y, z):
        x0, y0 = center
        return np.sqrt((x - x0)**2 + (y - y0)**2) < radius

    return FreeformShape(contains, [2 * radius] * 2, center)


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
    return _cpp.TranslationalSymmetry(*lengths)
