"""Structural information and utilities"""
import functools
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import _cpp
from . import pltutils
from .lattice import Lattice
from .utils import with_defaults, rotate_axes
from .support.alias import AliasArray
from .support.fuzzy_set import FuzzySet
from .support.structure import AbstractSites, Sites
from .results import Structure, StructureMap

__all__ = ['Sites', 'System', 'plot_hoppings', 'plot_periodic_boundaries', 'plot_sites',
           'structure_plot_properties']


class _CppSites(AbstractSites):
    """Tailored to the internal C++ compressed sublattice representation"""
    def __init__(self, impl: _cpp.System):
        self._positions = impl.positions
        self._cs = impl.compressed_sublattices
        self._lattice = impl.lattice

    @property
    def x(self):
        return self._positions.x

    @property
    def y(self):
        return self._positions.y

    @property
    def z(self):
        return self._positions.z

    @property
    def ids(self):
        return AliasArray(self._cs.decompressed(), self._lattice.sub_name_map)

    def __getitem__(self, item):
        return Sites([v[item] for v in self.positions], self.ids[item])


class System(Structure):
    """Structural data of a tight-binding model

    Stores positions, sublattice and hopping IDs for all lattice sites.
    """
    def __init__(self, impl: _cpp.System):
        super().__init__(_CppSites(impl), impl.hopping_blocks, impl.boundaries)
        self.impl = impl

    def __getstate__(self):
        return self.impl

    def __setstate__(self, impl):
        self.__init__(impl)

    @property
    def lattice(self) -> Lattice:
        """:class:`.Lattice` specification"""
        return Lattice.from_impl(self.impl.lattice)

    @property
    def expanded_positions(self):
        """`positions` expanded to `hamiltonian_size` by replicating for each orbital"""
        return self.impl.expanded_positions

    @property
    def hamiltonian_size(self) -> int:
        """The size of the Hamiltonian matrix constructed from this system

        Takes into account the number of orbitals/spins at each lattice site 
        which makes `hamiltonian_size` >= `num_sites`.
        """
        return self.impl.hamiltonian_size

    def with_data(self, data) -> StructureMap:
        """Map some data to this system"""
        data = self.reduce_orbitals(data)
        return StructureMap(data, self._sites, self._hoppings, self._boundaries)

    def find_nearest(self, position, sublattice=""):
        """Find the index of the atom closest to the given position

        Parameters
        ----------
        position : array_like
            Where to look.
        sublattice : Optional[str]
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        int
        """
        return self.impl.find_nearest(position, sublattice)

    def to_hamiltonian_indices(self, system_idx):
        """Translate the given system index into its corresponding Hamiltonian indices
        
        System indices are always scalars and index a single (x, y, z) site position.
        For single-orbital models there is a 1:1 correspondence between system and 
        Hamiltonian indices. However, for multi-orbital models the Hamiltonian indices
        are 1D arrays with a size corresponding to the number of orbitals on the target
        site.

        Parameters
        ----------
        system_idx : int

        Returns
        -------
        array_like
        """
        return self.impl.to_hamiltonian_indices(system_idx)

    def reduce_orbitals(self, data):
        """Sum up the contributions of individual orbitals in the given data

        Takes a 1D array of `hamiltonian_size` and returns a 1D array of `num_sites` size
        where the multiple orbital data has been reduced per site.

        Parameters
        ----------
        data : array_like
            Must be 1D and the equal to the size of the Hamiltonian matrix

        Returns
        -------
        array_like
        """
        data = np.atleast_1d(data)
        if data.size == self.num_sites:
            return data
        if data.ndim != 1 or data.size != self.hamiltonian_size:
            raise RuntimeError("The given data does not match the Hamiltonian size")

        start = 0
        reduced_data = []
        cs = self.impl.compressed_sublattices

        for nsites, norb in zip(cs.site_counts, cs.orbital_counts):
            end = start + nsites * norb
            reduced_data.append(data[start:end].reshape((-1, norb)).sum(axis=1))
            start = end

        return np.concatenate(reduced_data)


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
    # Also accept the plural form (`sites` and `hoppings`) since
    # this is a common mistake and source of frustration.
    site = kwargs.pop("sites", site)
    hopping = kwargs.pop("hoppings", hopping)

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
    plt.gca().autoscale_view()
    plt.xlabel("{} (nm)".format(axes[0]))
    plt.ylabel("{} (nm)".format(axes[1]))
    if add_margin:
        pltutils.set_min_axis_length(0.5)
        pltutils.set_min_axis_ratio(0.4)
        pltutils.despine(trim=True)
        pltutils.add_margin()
    else:
        pltutils.despine()


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

    rotate = functools.partial(rotate_axes, axes=axes)
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
    hoppings = hoppings.tocoo()
    if width == 0 or hoppings.data.size == 0:
        return

    kwargs = with_defaults(kwargs, zorder=-1)
    num_unique_hoppings = hoppings.data.max() + 1

    if "cmap" in kwargs:
        cmap = kwargs["cmap"]
        if cmap == 'auto':
            cmap = ['#666666', '#1b9e77', '#e6ab02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

        # create colormap from discrete colors
        if isinstance(cmap, (list, tuple)):
            unique_hop_ids = np.arange(num_unique_hoppings)
            kwargs['cmap'], kwargs['norm'] = pltutils.direct_cmap_norm(unique_hop_ids, cmap, blend)
        else:
            kwargs['cmap'] = cmap
    else:
        color = pltutils.blend_colors(color, "white", blend)

    rotate = functools.partial(rotate_axes, axes=axes)
    positions, offset = map(rotate, (positions, offset))

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
        lines = np.stack([pos[hoppings.row], pos[hoppings.col]], axis=1)
    else:
        sign, shift = boundary
        shift = rotate(shift)[:ndims]
        if sign > 0:
            lines = np.stack([pos[hoppings.row] + shift, pos[hoppings.col]], axis=1)
        else:
            lines = np.stack([pos[hoppings.row], pos[hoppings.col] - shift], axis=1)

    if ndims == 2:
        col = LineCollection(lines, colors=color, **kwargs)
        if "cmap" in kwargs:
            col.set_array(hoppings.data)
        ax.add_collection(col)

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
        col = Line3DCollection(lines, colors=color, lw=width, **kwargs)
        if "cmap" in kwargs:
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
        for s in shift_set:
            plot_sites(positions, data, offset=s, **{"blend": blend, **props["site"]})
            plot_hoppings(positions, hoppings, offset=s, **{"blend": blend, **props["hopping"]})

    # periodic boundary hoppings
    for level, blend in enumerate(blend_gradient, start=1):
        shift_set = _make_shift_set(boundaries, level)
        prev_shift_set = _make_shift_set(boundaries, level - 1)
        boundary_set = itertools.product(shift_set + prev_shift_set, (1, -1), boundaries)

        for shift, sign, boundary in boundary_set:
            if (shift + sign * boundary.shift) not in prev_shift_set:
                continue  # skip existing

            plot_hoppings(positions, boundary.hoppings.tocoo(), offset=shift,
                          boundary=(sign, boundary.shift), **{"blend": blend, **props["boundary"]})


def plot_site_indices(system):
    """Show the Hamiltonian index next to each atom (mainly for debugging)

    Parameters
    ----------
    system : System
    """
    for i, xy in enumerate(zip(system.x, system.y)):
        pltutils.annotate_box(i, xy)


def plot_hopping_values(system):
    """Show the hopping energy over each hopping line (mainly for debugging)

    Parameters
    ----------
    system : System
    """
    pos = system.xyz[:, :2]

    def get_energy(hopping_id):
        inv_name_map = {hop.family_id: name for name, hop in system.lattice.hoppings.items()}
        return inv_name_map[hopping_id]

    coo = system.hoppings.tocoo()
    for i, j, k in zip(coo.row, coo.col, coo.data):
        pltutils.annotate_box(get_energy(k), (pos[i] + pos[j]) / 2)

    for boundary in system.boundaries:
        coo = boundary.hoppings.tocoo()
        for i, j, k in zip(coo.row, coo.col, coo.data):
            pltutils.annotate_box(get_energy(k), (pos[i] + pos[j] + boundary.shift[:2]) / 2)
            pltutils.annotate_box(get_energy(k), (pos[i] + pos[j] - boundary.shift[:2]) / 2)
