"""Structural information and utilities"""
import functools
import itertools
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from scipy.sparse import csr_matrix

from . import _cpp
from . import pltutils
from .lattice import Lattice
from .utils import with_defaults
from .support.pickle import pickleable
from .support.fuzzy_set import FuzzySet
from .support.alias import AliasArray, AliasCSRMatrix

__all__ = ['Sites', 'System', 'plot_hoppings', 'plot_periodic_boundaries', 'plot_sites',
           'structure_plot_properties']


Positions = namedtuple('Positions', 'x y z')
# noinspection PyUnresolvedReferences
Positions.__doc__ = """
Named tuple of arrays

Attributes
----------
x, y, z : array_like
    1D arrays of Cartesian coordinates
"""


class Sites:
    """Utility class which stores site positions and sublattice IDs

    Attributes
    ----------
    x, y, z : np.ndarray
    sublattices : np.ndarray
    """
    def __init__(self, positions, sublattices):
        self.x, self.y, self.z = map(np.atleast_1d, positions)
        self.sublattices = np.atleast_1d(sublattices)

    @property
    def positions(self):
        """Named tuple of x, y, z positions"""
        return Positions(self.x, self.y, self.z)

    @property
    def sub_id(self) -> np.ndarray:
        """Alias for :attr:`sublattices`"""
        return self.sublattices

    def distances(self, target_position):
        """Return the distances of all sites from the target position

        Parameters
        ----------
        target_position : array_like

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> np.allclose(sites.distances([1, 0, 0]), [1, 0, 0.1])
        True
        """
        target_position = np.atleast_1d(target_position)
        ndim = len(target_position)
        positions = np.stack(self.positions[:ndim], axis=1)
        return np.linalg.norm(positions - target_position, axis=1)

    def find_nearest(self, target_position, target_sublattice=None):
        """Return the index of the position nearest the target

        Parameters
        ----------
        target_position : array_like
        target_sublattice : int
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        int

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> sites.find_nearest([1, 0, 0])
        1
        >>> sites.find_nearest([1, 0, 0], target_sublattice=0)
        2
        """
        distances = self.distances(target_position)
        if target_sublattice is None:
            return np.argmin(distances)
        else:
            return ma.argmin(ma.array(distances, mask=(self.sublattices != target_sublattice)))

    def argsort_nearest(self, target_position, target_sublattice=None):
        """Return an ndarray of site indices, sorted by distance from the target

        Parameters
        ----------
        target_position : array_like
        target_sublattice : int
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> sites = Sites(([0, 1, 1.1], [0, 0, 0], [0, 0, 0]), [0, 1, 0])
        >>> np.all(sites.argsort_nearest([1, 0, 0]) == [1, 2, 0])
        True
        >>> np.all(sites.argsort_nearest([1, 0, 0], target_sublattice=0) == [2, 0, 1])
        True
        """
        distances = self.distances(target_position)
        if target_sublattice is None:
            return np.argsort(distances)
        else:
            return ma.argsort(ma.array(distances, mask=(self.sublattices != target_sublattice)))


@pickleable(version=1)
class System:
    """Structural data of a tight-binding model

    Stores positions, sublattice and hopping IDs for all lattice sites.
    """
    def __init__(self, impl: _cpp.System):
        self.impl = impl

    @property
    def lattice(self) -> Lattice:
        """:class:`.Lattice` specification"""
        return self.impl.lattice

    @property
    def num_sites(self) -> int:
        """Total number of sites in the system"""
        return self.x.size

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
        return AliasArray(self.impl.sublattices, self.lattice.sub_name_map)

    @property
    def hoppings(self) -> csr_matrix:
        """Sparse matrix of hopping IDs"""
        return AliasCSRMatrix(self.impl.hoppings, mapping=self.lattice.hop_name_map)

    @property
    def boundaries(self) -> list:
        """List of :class:`.Boundary`"""
        return self.impl.boundaries

    def find_nearest(self, position, at_sublattice=None):
        """Find the index of the atom closest to the given position

        Parameters
        ----------
        position : array_like
            Where to look.
        at_sublattice : Optional[int]
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        int
        """
        at_sublattice = self.lattice[at_sublattice] if at_sublattice is not None else -1
        if hasattr(self.impl, 'find_nearest'):
            # use cpp implementation
            return self.impl.find_nearest(position, int(at_sublattice))
        else:
            # fallback numpy implementation
            sites = Sites(self.positions, self.sublattices)
            return sites.find_nearest(position, at_sublattice)

    def plot(self, num_periods=1, **kwargs):
        """Plot the structure: sites, hoppings and periodic boundaries (if any)

        Parameters
        ----------
        num_periods : int
            Number of times to repeat the periodic boundaries.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        props = structure_plot_properties(**kwargs)
        props['site'].setdefault('radius', self.lattice.site_radius_for_plot())

        plot_hoppings(self.positions, self.hoppings, **props['hopping'])
        plot_sites(self.positions, self.sublattices, **props['site'])
        plot_periodic_boundaries(self.positions, self.hoppings, self.boundaries,
                                 self.sublattices, num_periods, **props)

        decorate_structure_plot(**props)


def structure_plot_properties(axes='xyz', site=None, hopping=None, boundary=None, **kwargs):
    """Process structure plot properties

    Parameters
    ----------
    axes : str
        The spatial axes to plot. E.g. 'xy' for the default view,
        or 'yz', 'xz' and similar to plot a rotated view.
    site : dict
        Arguments forwarded to :func:`plot_sites`.
    hopping : dict
        Arguments forwarded to :func:`plot_hoppings`.
    boundary : dict
        Arguments forwarded to :func:`plot_periodic_boundaries`.
    **kwargs
        Additional args are reserved for internal implementation.

    Returns
    -------
    dict
    """
    invalid_args = kwargs.keys() - {'add_margin'}
    if invalid_args:
        raise RuntimeError("Invalid arguments: {}".format(','.join(invalid_args)))

    props = {'axes': axes, 'add_margin': kwargs.get('add_margin', True),
             'site': with_defaults(site, axes=axes),
             'hopping': with_defaults(hopping, axes=axes)}
    props['boundary'] = with_defaults(boundary, props['hopping'], color='#f40a0c')
    return props


def decorate_structure_plot(axes='xy', add_margin=True, **_):
    plt.gca().set_aspect('equal')
    plt.xlabel("{} (nm)".format(axes[0]))
    plt.ylabel("{} (nm)".format(axes[1]))
    if add_margin:
        pltutils.set_min_axis_length(0.5)
        pltutils.set_min_axis_ratio(0.4)
        pltutils.despine(trim=True)
        pltutils.add_margin()
    else:
        pltutils.despine()


def _rotate(position, axes):
    """Rotate axes in position"""
    missing_axes = set('xyz') - set(axes)
    for a in missing_axes:
        axes += a
    assert len(axes) == 3

    mapping = dict(x=0, y=1, z=2)
    return tuple(position[mapping[a]] for a in axes)


def _data_units_to_points(ax, value):
    """Convert a value from data units to points"""
    fig = ax.get_figure()
    length = fig.bbox_inches.width * ax.get_position().width
    length *= 72  # convert to points
    data_range = np.diff(ax.get_xlim())
    return value * (length / data_range)


def plot_sites(positions, data, radius=0.025, offset=(0, 0, 0), blend=1.0,
               cmap='auto', axes='xyz', **kwargs):
    """Plot circles at lattice site `positions` with colors based on `data`

    Parameters
    ----------
    positions : Tuple[array_like, array_like, array_like]
        Site coordinates in the form of an (x, y, z) tuple of 1D arrays.
    data : array_like
        Color data at each site. Should be a 1D array of the same size as `positions`.
        If the data is discrete with few unique values, the discrete `colors` parameter
        should be used. For continuous data, setting a `cmap` (colormap) is preferred.
    radius : Union[float, array_like]
        Radius (in data units) of the plotted circles representing lattice sites.
        Should be a scalar value or an array with the same size as `positions`.
    offset : Tuple[float, float, float]
        Offset all positions by a constant value.
    blend : float
        Blend all colors to white (fake alpha blending): expected values between 0 and 1.
    cmap : Union[str, List[str]]
        Either a regular matplotlib colormap or a list of discrete colors to apply to the
        drawn circles. In the latter case, it is assumed that `data` is discrete with only
        a few unique values. For example, sublattice data for graphene will only contain two
        unique values for the A and B sublattices which will be assigned the first two colors
        from the `cmap` list. For continuous data, a regular matplotlib colormap should be
        used instead.
    axes : str
        The spatial axes to plot. E.g. 'xy', 'yz', etc.
    **kwargs
        Forwarded to :class:`matplotlib.collections.CircleCollection`.

    Returns
    -------
    :class:`matplotlib.collections.CircleCollection`
    """
    if np.all(radius == 0):
        return

    kwargs = with_defaults(kwargs, alpha=0.97, lw=0.2, edgecolor=str(1-blend))

    if cmap == 'auto':
        cmap = ['#377ec8', '#ff7f00', '#41ae76', '#e41a1c',
                '#984ea3', '#ffff00', '#a65628', '#f781bf']
    elif cmap == 'pairs':
        cmap = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']

    # create colormap from discrete colors
    if isinstance(cmap, (list, tuple)):
        kwargs['cmap'], kwargs['norm'] = pltutils.direct_cmap_norm(data, cmap, blend)
    else:
        kwargs['cmap'] = cmap

    rotate = functools.partial(_rotate, axes=axes)
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

        def dynamic_scale(active_ax):
            """Rescale the circumference line width and radius based on data units"""
            scale = _data_units_to_points(active_ax, 0.005)  # [nm] reference for 1 screen point
            line_scale = np.clip(scale, 0.2, 1.1)  # don't make the line too thin or thick
            col.set_linewidth(line_scale * kwargs['lw'])
            if np.isscalar(radius):
                scale = _data_units_to_points(active_ax, 0.01)  # [nm]
                radius_scale = np.clip(2 - scale, 0.85, 1.3)
                col.radius = radius_scale * np.atleast_1d(radius)

        dynamic_scale(ax)
        ax.callbacks.connect('xlim_changed', dynamic_scale)
        ax.callbacks.connect('ylim_changed', dynamic_scale)
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


def plot_hoppings(positions, hoppings, width=1.0, offset=(0, 0, 0), blend=1.0, color='#666666',
                  axes='xyz', boundary=(), draw_only=(), **kwargs):
    """Plot lines between lattice sites at `positions` based on the `hoppings` matrix

    Parameters
    ----------
    positions : Tuple[array_like, array_like, array_like]
        Site coordinates in the form of an (x, y, z) tuple of 1D arrays.
    hoppings : :class:`~scipy.sparse.coo_matrix`
        Sparse matrix with the hopping data, usually :attr:`System.hoppings`.
        The `row` and `col` indices of the sparse matrix are used to draw lines between
        lattice sites, while `data` determines the color.
    width : float
        Width of the hopping plot lines.
    offset : Tuple[float, float, float]
        Offset all positions by a constant value.
    blend : float
        Blend all colors to white (fake alpha blending): expected values between 0 and 1.
    axes : str
        The spatial axes to plot. E.g. 'xy', 'yz', etc.
    color : str
        Set the same color for all hopping lines. To assign a different color for each
        hopping ID, use the `cmap` parameter.
    boundary : Tuple[int, array_like]
        If given, apply the boundary (sign, shift).
    draw_only : Iterable[str]
        Only draw lines for the hoppings named in this list.
    **kwargs
        Forwarded to :class:`matplotlib.collections.LineCollection`.

    Returns
    -------
    :class:`matplotlib.collections.LineCollection`
    """
    if width == 0 or hoppings.data.size == 0:
        return

    kwargs = with_defaults(kwargs, zorder=-1)

    cmap = kwargs.get('cmap', [color])
    if cmap == 'auto':
        cmap = ['#666666', '#1b9e77', '#e6ab02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

    # create colormap from discrete colors
    if isinstance(cmap, (list, tuple)):
        unique_hop_ids = np.arange(hoppings.data.max() + 1)
        kwargs['cmap'], kwargs['norm'] = pltutils.direct_cmap_norm(unique_hop_ids, cmap, blend)
    else:
        kwargs['cmap'] = cmap

    rotate = functools.partial(_rotate, axes=axes)
    positions, offset = map(rotate, (positions, offset))
    hoppings = hoppings.tocoo()

    # leave only the desired hoppings
    if draw_only:
        keep = np.zeros_like(hoppings.data, dtype=np.bool)
        for hop_id in draw_only:
            keep = np.logical_or(keep, hoppings.data == hop_id)
        hoppings.data = hoppings.data[keep]
        hoppings.col = hoppings.col[keep]
        hoppings.row = hoppings.row[keep]

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

        col = LineCollection(lines, **kwargs)
        col.set_array(hoppings.data)
        ax.add_collection(col)
        ax.autoscale_view()

        def dynamic_scale(active_ax):
            """Rescale the line width based on data units"""
            scale = _data_units_to_points(active_ax, 0.005)  # [nm] reference for 1 screen point
            scale = np.clip(scale, 0.6, 1.2)  # don't make the line too thin or thick
            col.set_linewidth(scale * width)

        dynamic_scale(ax)
        ax.callbacks.connect('xlim_changed', dynamic_scale)
        ax.callbacks.connect('ylim_changed', dynamic_scale)
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


def _make_shift_set(boundaries, level):
    """Return a set of boundary shift combinations for the given repetition level"""
    if level == 0:
        return FuzzySet([np.zeros(3)])

    base_shifts = [b.shift for b in boundaries] + [-b.shift for b in boundaries]
    all_shifts = (sum(c) for c in itertools.combinations_with_replacement(base_shifts, level))

    blacklist = sum(_make_shift_set(boundaries, l) for l in range(level))
    exclusive_shifts = (s for s in all_shifts if s not in blacklist)
    return FuzzySet(exclusive_shifts)


def plot_periodic_boundaries(positions, hoppings, boundaries, data, num_periods=1, **kwargs):
    """Plot the periodic boundaries of a system

    Parameters
    ----------
    positions : Tuple[array_like, array_like, array_like]
        Site coordinates in the form of an (x, y, z) tuple of 1D arrays.
    hoppings : :class:`~scipy.sparse.coo_matrix`
        Sparse matrix with the hopping data, usually :meth:`System.hoppings`.
        The `row` and `col` indices of the sparse matrix are used to draw lines between
        lattice sites, while `data` determines the color.
    boundaries : List[Boundary]
        Periodic boundaries of a :class:`System`.
    data : array_like
        Color data at each site. Should be a 1D array of the same size as `positions`.
    num_periods : int
        Number of times to repeat the periodic boundaries.
    **kwargs
        Additional plot arguments as specified in :func:`.structure_plot_properties`.
    """
    props = structure_plot_properties(**kwargs)

    # the periodic parts will fade out gradually at each level of repetition
    blend_gradient = np.linspace(0.5, 0.15, num_periods)

    # periodic unit cells
    for level, blend in enumerate(blend_gradient, start=1):
        shift_set = _make_shift_set(boundaries, level)
        for shift in shift_set:
            plot_sites(positions, data, offset=shift, blend=blend, **props['site'])
            plot_hoppings(positions, hoppings, offset=shift, blend=blend, **props['hopping'])

    # periodic boundary hoppings
    for level, blend in enumerate(blend_gradient, start=1):
        shift_set = _make_shift_set(boundaries, level)
        prev_shift_set = _make_shift_set(boundaries, level - 1)
        boundary_set = itertools.product(shift_set + prev_shift_set, (1, -1), boundaries)

        for shift, sign, boundary in boundary_set:
            if (shift + sign * boundary.shift) not in prev_shift_set:
                continue  # skip existing

            plot_hoppings(positions, boundary.hoppings.tocoo(), offset=shift, blend=blend,
                          boundary=(sign, boundary.shift), **props['boundary'])


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
