import functools
import itertools
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import with_defaults
from .support.sparse import SparseMatrix
from .support.pickle import pickleable
from .support.fuzzy_set import FuzzySet

__all__ = ['Positions', 'Boundary', 'System', 'plot_hoppings', 'plot_sites']


Positions = namedtuple('Positions', 'x y z')
# noinspection PyUnresolvedReferences
Positions.__doc__ = """
Named tuple of arrays

Attributes
----------
x, y, z : array_like
    1D arrays of Cartesian coordinates
"""


@pickleable(impl='shift hoppings.')
class Boundary:
    """Periodic boundary"""

    def __init__(self, impl: _cpp.Boundary):
        self.impl = impl

    @property
    def shift(self) -> np.ndarray:
        """Position shift of the periodic boundary condition"""
        return self.impl.shift

    @property
    def hoppings(self) -> SparseMatrix:
        """Sparse matrix of the boundary hoppings"""
        return SparseMatrix(self.impl.hoppings)


@pickleable(impl='shift indices inner_hoppings. outer_hoppings.')
class Port:
    """Port for a lead to plug into"""

    def __init__(self, impl: _cpp.Port):
        self.impl = impl

    @property
    def shift(self) -> np.ndarray:
        """Position shift of the periodic boundary condition"""
        return self.impl.shift

    @property
    def indices(self) -> np.ndarray:
        """Map of lead indices to main system indices"""
        return np.array(self.impl.indices)

    @property
    def inner_hoppings(self) -> SparseMatrix:
        """Sparse matrix of the inner lead hoppings"""
        return SparseMatrix(self.impl.inner_hoppings)

    @property
    def outer_hoppings(self) -> SparseMatrix:
        """Sparse matrix of the outer (boundary) hoppings"""
        return SparseMatrix(self.impl.outer_hoppings)


@pickleable(impl='positions sublattices hoppings. boundaries[] ports[]')
class System:
    """Structural data of the model

    Stores positions, sublattice and hopping IDs for all lattice sites.
    """

    def __init__(self, impl: _cpp.System):
        self.impl = impl

    @property
    def num_sites(self) -> int:
        """Total number of sites in the system"""
        return self.impl.num_sites

    @property
    def x(self) -> np.ndarray:
        """1D array of x coordinates"""
        return self.impl.positions.x

    @property
    def y(self) -> np.ndarray:
        """1D array of y coordinates"""
        return self.impl.positions.y

    @property
    def z(self) -> np.ndarray:
        """1D array of z coordinates"""
        return self.impl.positions.z

    @property
    def xyz(self) -> np.ndarray:
        """Return a new array with shape=(N, 3). Convenient, but slow for big systems."""
        return np.array(self.positions).T

    @property
    def positions(self):
        """Named tuple of x, y, z positions"""
        return Positions(self.x, self.y, self.z)

    @property
    def sublattices(self) -> np.ndarray:
        """1D array of sublattice IDs"""
        return self.impl.sublattices

    @property
    def hoppings(self) -> SparseMatrix:
        """Sparse matrix of hopping IDs"""
        return SparseMatrix(self.impl.hoppings)

    @property
    def boundaries(self):
        """List of :class:`.Boundary`"""
        return [Boundary(b) for b in self.impl.boundaries]

    @property
    def ports(self):
        """List of :class:`.Port`"""
        return [Port(p) for p in self.impl.ports]

    def find_nearest(self, position, at_sublattice=-1):
        """Find the index of the atom closest to the given position

        Parameters
        ----------
        position : array_like
            Where to look.
        at_sublattice : int
            Look for a specific sublattice site, or -1 if any will do (default).

        Returns
        -------
        int
            Index of the site or -1 if not found.
        """
        if hasattr(self.impl, 'find_nearest'):
            # use cpp implementation
            return self.impl.find_nearest(position, int(at_sublattice))
        else:
            # fallback numpy implementation
            r = np.array(position)
            distance = np.linalg.norm(self.xyz[:, :len(r)] - r, axis=1)
            if at_sublattice < 0:
                return np.argmin(distance)
            else:
                from numpy import ma
                masked_distance = ma.array(distance, mask=self.sublattices != at_sublattice)
                return ma.argmin(masked_distance)

    def plot(self, site_radius=0.025, hopping_width=1.0, num_periods=1, lead_length=6, axes='xy',
             site_props=None, hopping_props=None, boundary_props=None):
        """Plot the structure of the system: sites and hoppings

        Parameters
        ----------
        site_radius : float
            Radius (in data units) of the circle representing a lattice site.
        hopping_width : float
            Width (in figure units) of the hopping lines.
        num_periods : int
            Number of times to repeat the periodic boundaries.
        lead_length : int
            Number of times to repeat the lead structure.
        axes : str
            The spatial axes to plot. E.g. 'xy', 'yz', etc.
        site_props : Optional[dict]
            Forwarded to :class:`.CircleCollection`: additional site plotting options.
        hopping_props : Optional[dict]
            Forwarded to :class:`.LineCollection`: additional hopping line options.
        boundary_props : Optional[dict]
            Forwarded to :class:`.LineCollection`: additional boundary hopping line options.
        """
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel("{} (nm)".format(axes[0]))
        ax.set_ylabel("{} (nm)".format(axes[1]))

        site_props = with_defaults(site_props, axes=axes)
        hopping_props = with_defaults(hopping_props, axes=axes)
        boundary_props = with_defaults(boundary_props, hopping_props, colors='#d40a0c')

        plot_hoppings(self.positions, self.hoppings.tocoo(), hopping_width, **hopping_props)
        plot_sites(self.positions, self.sublattices, site_radius, **site_props)

        plot_periodic_structure(self.positions, self.hoppings.tocoo(), self.boundaries,
                                self.sublattices, site_radius, hopping_width, num_periods,
                                site_props, hopping_props, boundary_props)

        properties = dict(site=site_props, hopping=hopping_props, boundary=boundary_props)
        plot_lead_structure(self, site_radius, hopping_width, lead_length, **properties)

        pltutils.set_min_axis_length(0.5)
        pltutils.despine(trim=True)
        pltutils.add_margin()


def _rotate(position, axes):
    """Rotate axes in position"""
    missing_axes = set('xyz') - set(axes)
    for a in missing_axes:
        axes += a
    assert len(axes) == 3

    mapping = dict(x=0, y=1, z=2)
    return tuple(position[mapping[a]] for a in axes)


def plot_hoppings(positions, hoppings, width, offset=(0, 0, 0), blend=1.0, boundary=(), **kwargs):
    """Plot hopping lines between sites

    Parameters
    ----------
    positions : Positions
        Site coordinates in the form of a (x, y, z) tuple of arrays.
    hoppings : :class:`~scipy.sparse.coo_matrix`
        Sparse COO matrix with the hopping data, usually `model.system.hoppings.tocoo()`.
    width : float
        Width of the hopping plot lines.
    offset : Tuple[float, float, float]
        Offset all positions by a constant value.
    blend : float
        Blend all colors to white (fake alpha blending): expected values between 0 and 1.
    boundary : Tuple[int, array_like]
        If given, apply the boundary (sign, shift).
    **kwargs
        Forwarded to matplotlib's :class:`.LineCollection`.

    Returns
    -------
    :class:`.LineCollection`
    """
    if width == 0 or hoppings.data.size == 0:
        return

    kwargs = with_defaults(kwargs, zorder=-1, colors='#666666')

    colors = kwargs.pop('colors')
    if colors == 'default':
        colors = ["#666666", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    unique_hop_ids = np.arange(hoppings.data.max() + 1)
    kwargs['cmap'], kwargs['norm'] = pltutils.direct_cmap_norm(unique_hop_ids, colors, blend)

    rotate = functools.partial(_rotate, axes=kwargs.pop('axes', 'xyz'))
    positions, offset = map(rotate, (positions, offset))

    ax = plt.gca()
    ndims = 3 if ax.name == '3d' else 2
    pos = np.array(positions[:ndims]).T + np.array(offset[:ndims])

    if not boundary:
        lines = ((pos[i], pos[j]) for i, j in zip(hoppings.row, hoppings.col))
    else:
        sign, shift = boundary
        shift = rotate(shift)[:ndims]
        if sign > 0:
            lines = ((pos[i] + shift, pos[j]) for i, j in zip(hoppings.row, hoppings.col))
        else:
            lines = ((pos[i], pos[j] - shift) for i, j in zip(hoppings.row, hoppings.col))

    if ndims == 2:
        from matplotlib.collections import LineCollection
        col = LineCollection(lines, lw=width, **kwargs)
        col.set_array(hoppings.data)
        ax.add_collection(col)
        ax.autoscale_view()
    else:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        had_data = ax.has_data()
        col = Line3DCollection(list(lines), lw=width, **kwargs)
        col.set_array(hoppings.data)
        ax.add_collection3d(col)

        ax.set_zmargin(0.5)
        minmax = np.vstack((pos.min(axis=0), pos.max(axis=0))).T
        ax.auto_scale_xyz(*minmax, had_data=had_data)

    return col


def plot_sites(positions, data, radius, offset=(0, 0, 0), blend=1.0, **kwargs):
    """Plot circles at lattice site positions

    Parameters
    ----------
    positions : Positions
        Site coordinates in the form of a (x, y, z) tuple of arrays.
    data : array_like
        Color data at each site. Should be a 1D array of the same size as `positions`.
    radius : Union[float, array_like]
        Radius (in data units) of the plotted circles representing lattice sites.
        Should be a scalar value or array with the same size as `positions`.
    offset : Tuple[float, float, float]
            Offset all positions by a constant value.
    blend : float
        Blend all colors to white (fake alpha blending): expected values between 0 and 1.
    **kwargs
        Forwarded to :class:`.CircleCollection`.

    Returns
    -------
    :class:`.CircleCollection`
    """
    if np.all(radius == 0):
        return

    kwargs = with_defaults(kwargs, alpha=0.97, lw=0.1, edgecolor=str(1-blend))

    # create colormap from discrete colors
    if 'cmap' not in kwargs:
        colors = kwargs.pop('colors', None)
        if not colors or colors == 'default':
            colors = ["#377ec8", "#ff7f00", "#41ae76", "#e41a1c",
                      "#984ea3", "#ffff00", "#a65628", "#f781bf"]
        elif colors == 'pairs':
            colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
                      "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
        kwargs['cmap'], kwargs['norm'] = pltutils.direct_cmap_norm(data, colors, blend)

    rotate = functools.partial(_rotate, axes=kwargs.pop('axes', 'xyz'))
    positions, offset = map(rotate, (positions, offset))

    # create array of (x, y) points
    points = np.array(positions[:2]).T + offset[:2]

    ax = plt.gca()
    if ax.name != '3d':
        # sort based on z position to get proper 2D z-order
        z = positions[2]
        if len(np.unique(z)) > 1:
            idx = z.argsort()
            if not np.isscalar(radius):
                radius = radius[idx]
            points, data = points[idx], data[idx]

        from pybinding.support.collections import CircleCollection
        col = CircleCollection(radius, offsets=points, transOffset=ax.transData, **kwargs)
        col.set_array(data)

        ax.add_collection(col)
        ax.autoscale_view()
    else:
        from pybinding.support.collections import Circle3DCollection
        col = Circle3DCollection(radius/8, offsets=points, transOffset=ax.transData, **kwargs)
        col.set_array(data)
        z = positions[2] + offset[2]
        col.set_3d_properties(z, 'z')

        had_data = ax.has_data()
        ax.add_collection(col)
        minmax = tuple((v.min(), v.max()) for v in positions)
        ax.auto_scale_xyz(*minmax, had_data=had_data)

    return col


def _make_shift_set(boundaries, level):
    """Return a set of boundary shift combinations for the given repetition level"""
    if level == 0:
        return FuzzySet([np.zeros(3)])

    base_shifts = [b.shift for b in boundaries] + [-b.shift for b in boundaries]
    all_shifts = (sum(c) for c in itertools.combinations_with_replacement(base_shifts, level))

    blacklist = sum(_make_shift_set(boundaries, l) for l in range(level))
    exclusive_shifts = (s for s in all_shifts if s not in blacklist)
    return FuzzySet(exclusive_shifts)


def plot_periodic_structure(positions, hoppings, boundaries, data, site_radius, hopping_width,
                            num_periods, site_props=None, hopping_props=None, boundary_props=None):
    site_props = site_props or {}
    hopping_props = hopping_props or {}
    boundary_props = with_defaults(boundary_props, hopping_props)

    # the periodic parts will fade out gradually at each level of repetition
    blend_gradient = np.linspace(0.5, 0.15, num_periods)

    # periodic unit cells
    for level, blend in enumerate(blend_gradient, start=1):
        shift_set = _make_shift_set(boundaries, level)
        for shift in shift_set:
            plot_sites(positions, data, site_radius, shift, blend, **site_props)
            plot_hoppings(positions, hoppings, hopping_width, shift, blend, **hopping_props)

    # periodic boundary hoppings
    for level, blend in enumerate(blend_gradient, start=1):
        shift_set = _make_shift_set(boundaries, level)
        prev_shift_set = _make_shift_set(boundaries, level - 1)
        boundary_set = itertools.product(shift_set + prev_shift_set, (1, -1), boundaries)

        for shift, sign, boundary in boundary_set:
            if (shift + sign * boundary.shift) not in prev_shift_set:
                continue  # skip existing

            plot_hoppings(positions, boundary.hoppings.tocoo(), hopping_width * 1.4, shift, blend,
                          boundary=(sign, boundary.shift), **boundary_props)


def plot_lead_structure(system, site_radius, hopping_width, lead_length=6, **properties):
    """Plot the sites, hoppings and periodic boundaries of the systems leads

    Parameters
    ----------
    system : System
        Needs to have `positions`, `sublattices` and `ports` attributes.
    site_radius : float
        Radius (in data units) of the circle representing a lattice site.
    hopping_width : float
        Width (in figure units) of the hopping lines.
    lead_length : int
        Number of times to repeat the lead's periodic boundaries.
    **properties
        Site, hopping and boundary properties: to be forwarded to their respective plots.
    """
    blend_gradient = np.linspace(0.5, 0.1, lead_length)
    for port in system.ports:
        for i, blend in enumerate(blend_gradient, start=1):
            pos = tuple(v[port.indices] for v in system.positions)
            sub = system.sublattices[port.indices]
            offset = i * port.shift

            plot_sites(pos, sub, site_radius, offset, blend, **properties.get('site', {}))
            plot_hoppings(pos, port.inner_hoppings.tocoo(), hopping_width, offset, blend,
                          **properties.get('hopping', {}))
            plot_hoppings(pos, port.outer_hoppings.tocoo(), hopping_width * 1.6, offset, blend,
                          boundary=(1, -port.shift), **properties.get('boundary', {}))


def plot_site_indices(system):
    """Show the Hamiltonian index next to each atom (mainly for debugging)

    Parameters
    ----------
    system : System
    """
    for i, xy in enumerate(zip(system.x, system.y)):
        pltutils.annotate_box(i, xy)


def plot_hopping_values(system, lattice):
    """Show the hopping energy over each hopping line (mainly for debugging)

    Parameters
    ----------
    system : System
    lattice : Lattice
    """
    pos = system.xyz[:, :2]

    def get_energy(hopping_id):
        t = lattice.hopping_energies[hopping_id]
        return t.real if t.imag == 0 else t

    for i, j, k in system.hoppings.triplets():
        pltutils.annotate_box(get_energy(k), (pos[i] + pos[j]) / 2)

    for boundary in system.boundaries:
        for i, j, k in boundary.hoppings.triplets():
            pltutils.annotate_box(get_energy(k), (pos[i] + pos[j] + boundary.shift[:2]) / 2)
            pltutils.annotate_box(get_energy(k), (pos[i] + pos[j] - boundary.shift[:2]) / 2)
