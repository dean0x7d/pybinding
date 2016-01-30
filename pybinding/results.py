"""Processing and presentation of computed data

Result objects hold computed data and offer postprocessing and plotting functions
which are specifically adapted to the nature of the stored data.
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from . import pltutils
from .utils import with_defaults, x_pi
from .system import Positions, plot_sites, plot_hoppings, plot_periodic_structure
from .support.pickle import pickleable

__all__ = ['make_path', 'DOS', 'LDOS', 'SpatialMap', 'StructureMap',
           'Eigenvalues', 'Bands', 'Sweep', 'NDSweep']


def make_path(k0, k1, *ks, step=0.1):
    """Create a path which connects the given k points.

    Parameters
    ----------
    k0, k1, *ks
        Points in k-space to connect.
    step : float
        Length in k-space between two samples.

    Examples
    --------
    >>> np.allclose(make_path(0, 3, -1, step=1).T, [0, 1, 2, 3, 2, 1, 0, -1])
    True
    >>> np.allclose(make_path([0, 0], [2, 3], [-1, 4], step=1.4),
    ...             [[0, 0], [1, 1.5], [2, 3], [0.5, 3.5], [-1, 4]])
    True
    """
    k_points = [np.atleast_1d(k) for k in (k0, k1) + ks]

    k_paths = []
    for k_start, k_end in zip(k_points[:-1], k_points[1:]):
        num_steps = np.linalg.norm(k_end - k_start) // step
        # k_path.shape == num_steps, k_space_dimensions
        k_path = np.array([np.linspace(s, e, num_steps, endpoint=False)
                           for s, e in zip(k_start, k_end)]).T
        k_paths.append(k_path)
    k_paths.append(k_points[-1])

    return np.vstack(k_paths)


@pickleable
class DOS:
    """Density of states as a function of energy

    Attributes
    ----------
    energy : array_like
    dos : array_like
    """
    def __init__(self, energy, dos):
        self.energy = np.atleast_1d(energy)
        self.dos = np.atleast_1d(dos)

    def plot(self, **kwargs):
        """Labeled line plot

        Parameters
        ----------
        **kwargs
            Forwarded to `plt.plot()`.
        """
        plt.plot(self.energy, self.dos, **kwargs)
        plt.xlim(self.energy.min(), self.energy.max())
        plt.ylabel('DOS')
        plt.xlabel('E (eV)')
        pltutils.despine()


@pickleable
class LDOS:
    """Local density of states as a function of energy

    Attributes
    ----------
    energy : array_like
    ldos : array_like
    """
    def __init__(self, energy, ldos):
        self.energy = np.atleast_1d(energy)
        self.ldos = np.atleast_1d(ldos)

    def plot(self, **kwargs):
        """Labeled line plot

        Parameters
        ----------
        **kwargs
            Forwarded to `plt.plot()`.
        """
        plt.plot(self.energy, self.ldos, **kwargs)
        plt.xlim(self.energy.min(), self.energy.max())
        plt.ylabel('LDOS')
        plt.xlabel('E (eV)')
        pltutils.despine()


@pickleable
class SpatialMap:
    """Represents some spatially dependent property: data mapped to site positions

    Attributes
    ----------
    data : np.ndarray
        1D array of values which correspond to x, y, z coordinates.
    pos : tuple of np.ndarray
        Lattice site positions. Named tuple with x, y, z fields, each a 1D array.
    sub : np.ndarray
        Sublattice ID for each position.
    """
    def __init__(self, data, positions, sublattices):
        self.data = np.atleast_1d(data)
        self.pos = Positions(*positions)  # maybe convert from tuple
        self.sub = np.atleast_1d(sublattices)

    @classmethod
    def from_system(cls, data, system):
        """Alternate constructor which takes `pos` and `sub` from `system`

        Parameters
        ----------
        data : np.ndarray
        system : :class:`.System`
        """
        return cls(data, system.positions, system.sublattices)

    def copy(self) -> 'SpatialMap':
        """Return a deep copy"""
        return deepcopy(self)

    def save_txt(self, filename):
        with open(filename + '.dat', 'w') as file:
            file.write('# {:12}{:13}{:13}\n'.format('x(nm)', 'y(nm)', 'data'))
            for x, y, d in zip(self.pos.x, self.pos.y, self.data):
                file.write(("{:13.5e}" * 3 + '\n').format(x, y, d))

    def filter(self, idx):
        """Leave only the sites indicated by `idx`: same rules as numpy indexing

        Parameters
        ----------
        idx : array_like
        """
        # TODO: consider making this class immutable and use __getitem__ instead of filter
        self.data = self.data[idx]
        self.pos = Positions(*map(lambda v: v[idx], self.pos))
        self.sub = self.sub[idx]

    def crop(self, **limits):
        """Leave only the sites that are within the given limits

        Parameters
        ----------
        **limits
            Attribute names and corresponding limits.

        Examples
        --------
        Leave only the data where -10 <= x < 10 and 2 <= y < 4::

            s.crop(x=(-10, 10), y=(2, 4))
        """
        idx = np.ones(self.pos.x.size, dtype=bool)
        attrib = {'x': self.pos.x, 'y': self.pos.y, 'z': self.pos.z,
                  'sub': self.sub, 'data': self.data}
        for key, limit in limits.items():
            if key not in attrib:
                raise AttributeError("'{}' not found in SpatialMap".format(key))

            v = attrib[key]
            idx = np.logical_and(idx, v >= limit[0])
            idx = np.logical_and(idx, v < limit[1])

        self.filter(idx)

    def clip(self, v_min, v_max):
        self.data = np.clip(self.data, v_min, v_max)

    def convolve(self, sigma=0.25):
        # TODO: slow and only works in the xy-plane
        x, y, _ = self.pos
        r = np.sqrt(x**2 + y**2)

        data = np.empty_like(self.data)
        for i in range(len(data)):
            idx = np.abs(r - r[i]) < sigma
            data[i] = np.sum(self.data[idx] * np.exp(-0.5 * ((r[i] - r[idx]) / sigma)**2))
            data[i] /= np.sum(np.exp(-0.5 * ((r[i] - r[idx]) / sigma)**2))

        self.data = data

    @staticmethod
    def _decorate_plot():
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        pltutils.despine(trim=True)

    def plot_pcolor(self, **kwargs):
        """Color plot of the xy plane

        Parameters
        ----------
        **kwargs
            Forwarded to `plt.tripcolor()`.
        """
        x, y, _ = self.pos
        kwargs = with_defaults(kwargs, shading='gouraud', rasterized=True)
        pcolor = plt.tripcolor(x, y, self.data, **kwargs)
        self._decorate_plot()
        return pcolor

    def plot_contourf(self, num_levels=50, **kwargs):
        """Filled contour plot of the xy plane

        Parameters
        ----------
        num_levels : int
            Number of contour levels.
        **kwargs
            Forwarded to `plt.tricontourf()`.
        """
        levels = np.linspace(self.data.min(), self.data.max(), num=num_levels)
        x, y, _ = self.pos
        kwargs = with_defaults(kwargs, levels=levels, rasterized=True)
        contourf = plt.tricontourf(x, y, self.data, **kwargs)
        self._decorate_plot()
        return contourf

    def plot_contour(self, **kwargs):
        """Contour plot of the xy plane

        Parameters
        ----------
        **kwargs
            Forwarded to `plt.tricontour()`.
        """
        x, y, _ = self.pos
        contour = plt.tricontour(x, y, self.data, **kwargs)
        self._decorate_plot()
        return contour


@pickleable
class StructureMap(SpatialMap):
    """A `SpatialMap` that also includes hoppings between sites

    Attributes
    ----------
    hoppings : scipy.csr_matrix
        Sparse matrix of hopping IDs. See `System.hoppings`.
    boundaries : list of scipy.csr_matrix
        Boundary hoppings. See 'System.boundaries`.
    """
    def __init__(self, data, positions, sublattices, hoppings, boundaries):
        super().__init__(data, positions, sublattices)
        self.hoppings = hoppings
        self.boundaries = boundaries

    @classmethod
    def from_system(cls, data, system):
        return cls(data, system.positions, system.sublattices,
                   system.hoppings.tocsr(), system.boundaries)

    def copy(self) -> 'StructureMap':
        """Return a deep copy"""
        return deepcopy(self)

    @property
    def spatial_map(self) -> SpatialMap:
        """Just the :class:`SpatialMap` subset without hoppings"""
        return SpatialMap(self.data, self.pos, self.sub)

    @staticmethod
    def _filter_csr_matrix(csr, idx):
        """Indexing must preserve all data, even zeros"""
        plus_one = csr_matrix((csr.data + 1, csr.indices, csr.indptr), csr.shape)
        plus_one = plus_one[idx][:, idx]
        return csr_matrix((plus_one.data - 1, plus_one.indices, plus_one.indptr), plus_one.shape)

    def filter(self, idx):
        super().filter(idx)
        self.hoppings = self._filter_csr_matrix(self.hoppings, idx)
        for boundary in self.boundaries:
            boundary.hoppings = self._filter_csr_matrix(boundary.hoppings, idx)

    def plot_structure(self, cmap='YlGnBu', site_radius=(0.03, 0.05), hopping_width=1,
                       num_periods=1, site_props=None, hopping_props=None, cbar_props=None):
        """Plot the spatial structure with a colormap of :attr:`data` at the lattice sites

        Both the site size and color are used to display the data.

        Parameters
        ----------
        cmap : str
            Matplotlib colormap to be used for the data.
        site_radius : Tuple[float, float]
            Min and max radius of lattice sites. This range will be used to visually
            represent the magnitude of the data.
        hopping_width : float
            Width of the lines representing the hoppings.
        num_periods : int
            Number of times to repeat periodic boundaries.
        site_props : dict
            Forwarded to :func:`.plot_sites`.
        hopping_props : dict
            Forwarded to :func:`.plot_hoppings`.
        cbar_props : dict
            Forwarded to `plt.colorbar`.
        """
        ax = plt.gca()
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        def to_radii(data):
            if not isinstance(site_radius, (tuple, list)):
                return site_radius

            positive_data = data - data.min()
            maximum = positive_data.max()
            if not np.allclose(maximum, 0):
                delta = site_radius[1] - site_radius[0]
                return site_radius[0] + delta * positive_data / maximum
            else:
                return site_radius[1]

        radii = to_radii(self.data)
        site_props = with_defaults(site_props, cmap=cmap)
        collection = plot_sites(self.pos, self.data, radii, **site_props)

        hop = self.hoppings.tocoo()
        hopping_props = with_defaults(hopping_props, colors='#bbbbbb')
        plot_hoppings(self.pos, hop, hopping_width, **hopping_props)

        site_props['alpha'] = hopping_props['alpha'] = 0.5
        plot_periodic_structure(self.pos, hop, self.boundaries, self.data, radii,
                                hopping_width, num_periods, site_props, hopping_props)

        if cbar_props is not False:
            pltutils.colorbar(collection, **with_defaults(cbar_props))
        pltutils.despine(trim=True)
        pltutils.add_margin()


@pickleable
class Eigenvalues:
    """Hamiltonian eigenvalues with optional probability map

    Attributes
    ----------
    values : np.ndarray
    probability : np.ndarray
    """
    def __init__(self, eigenvalues, probability=None):
        self.values = np.atleast_1d(eigenvalues)
        self.probability = np.atleast_1d(probability)

    @property
    def indices(self):
        return np.arange(0, self.values.size)

    def _decorate_plot(self, mark_degenerate, number_states):
        """Common elements for the two eigenvalue plots"""
        if mark_degenerate:
            # draw lines between degenerate states
            from .solver import Solver
            from matplotlib.collections import LineCollection
            pairs = ((s[0], s[-1]) for s in Solver.find_degenerate_states(self.values))
            lines = [[(i, self.values[i]) for i in pair] for pair in pairs]
            plt.gca().add_collection(LineCollection(lines, color='black', alpha=0.5))

        if number_states:
            # draw a number next to each state
            for index, energy in enumerate(self.values):
                pltutils.annotate_box(index, (index, energy), fontsize='x-small',
                                      xytext=(0, -10), textcoords='offset points')

        plt.xlabel('state')
        plt.ylabel('E (eV)')
        plt.xlim(-1, len(self.values))
        pltutils.despine(trim=True)

    def plot(self, mark_degenerate=True, show_indices=False, **kwargs):
        """Standard eigenvalues scatter plot

        Parameters
        ----------
        mark_degenerate : bool
            Plot a line which connects degenerate states.
        show_indices : bool
            Plot index number next to all states.
        **kwargs
            Forwarded to plt.scatter().
        """
        plt.scatter(self.indices, self.values, **with_defaults(kwargs, c='#377ec8', s=15, lw=0.1))
        self._decorate_plot(mark_degenerate, show_indices)

    def plot_heatmap(self, size=(7, 77), mark_degenerate=True, show_indices=False, **kwargs):
        """Eigenvalues scatter plot with a heatmap indicating probability density

        Parameters
        ----------
        size : Tuple[int, int]
            Min and max scatter dot size.
        mark_degenerate : bool
            Plot a line which connects degenerate states.
        show_indices : bool
            Plot index number next to all states.
        **kwargs
            Forwarded to plt.scatter().
        """
        if self.probability is None:
            return self.plot(mark_degenerate, show_indices, **kwargs)

        # higher probability states should be drawn above lower ones
        idx = np.argsort(self.probability)
        indices, energy, probability = (v[idx] for v in
                                        (self.indices, self.values, self.probability))

        scatter_point_sizes = size[0] + size[1] * probability / probability.max()
        plt.scatter(indices, energy, **with_defaults(kwargs, cmap='YlOrRd', lw=0.2, alpha=0.85,
                                                     c=probability, s=scatter_point_sizes))

        self._decorate_plot(mark_degenerate, show_indices)
        return self.probability.max()


@pickleable
class Bands:
    """Band structure along a path in k-space

    Attributes
    ----------
    k_points : List[Tuple[float]]
    k_path : array_like
    bands : array_like
    """
    def __init__(self, k_points, k_path, bands):
        self.k_points = k_points
        self.k_path = k_path
        self.bands = bands

    @staticmethod
    def _point_names(k_points):
        names = []
        for k_point in k_points:
            values = map(x_pi, k_point)
            fmt = "[{}]" if len(k_point) > 1 else "{}"
            names.append(fmt.format(', '.join(values)))
        return names

    def plot(self, point_labels=None, **kwargs):
        """Line plot of the band structure

        Parameters
        ----------
        point_labels : List[str]
            Labels for the `k_points`.
        **kwargs
            Forwarded to `plt.plot()`.
        """
        default_color = pltutils.get_palette('Set1')[1]
        plt.plot(self.bands, **with_defaults(kwargs, color=default_color))

        plt.xlim(0, len(self.bands) - 1)
        plt.xlabel('k-space')
        plt.ylabel('E (eV)')
        pltutils.add_margin()
        pltutils.despine(trim=True)

        border_indices = [idx for idx, k in enumerate(self.k_path)
                          if any(np.allclose(k, k0) for k0 in self.k_points)]

        if not point_labels:
            point_labels = self._point_names(self.k_points)
        plt.xticks(border_indices, point_labels)

        for idx in border_indices:
            ymax = plt.gca().transLimits.transform([0, max(self.bands[idx])])[1]
            plt.axvline(idx, ymax=ymax, color='0.4', ls=':', zorder=-1)

    def plot_kpath(self, point_labels=None, **kwargs):
        """Quiver plot of path in k-space along which the bands were computed

        Combine with :meth:`.Lattice.plot_brillouin_zone` to see the path in context.

        Parameters
        ----------
        point_labels : List[str]
            Labels for the `k_points`.
        **kwargs
            Forwarded to `plt.quiver()`.
        """
        ax = plt.gca()
        ax.set_aspect('equal')

        default_color = pltutils.get_palette('Set1')[1]
        kwargs = with_defaults(kwargs, scale_units='xy', angles='xy', scale=1, zorder=2,
                               lw=1.5, color=default_color, edgecolor=default_color)

        x, y = map(np.array, zip(*self.k_points))
        plt.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y), **kwargs)

        ax.autoscale_view()
        pltutils.add_margin(0.5)
        pltutils.despine(trim=True)

        if point_labels:
            for k_point, label in zip(self.k_points, point_labels):
                ha, va = pltutils.align(*(-k_point))
                pltutils.annotate_box(label, k_point * 1.05, fontsize='large',
                                      ha=ha, va=va, bbox=dict(lw=0))


@pickleable
class Sweep:
    """2D parameter sweep with 'x' and 'y' 1D array parameters and 'data' 2D array result.

    Attributes
    ----------
    x : np.ndarray
        1D array with x-axis values - usually the primary parameter being swept.
    y : np.ndarray
        1D array with y-axis values - usually the secondary parameter.
    data : np.ndarray
        2D array with shape == (x.size, y.size) containing the main result data.
    labels : dict
        Plot labels: 'title', 'x', 'y' and 'data'.
    tags : dict
        Any additional user defined variables.
    """
    def __init__(self, x, y, data, labels=None, tags=None):
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.data = np.atleast_2d(data)

        self.labels = with_defaults(labels, title="", x="x", y="y", data="data")
        self.tags = tags

    def copy(self) -> 'Sweep':
        return deepcopy(self)

    @property
    def plain_labels(self):
        """Labels dict with latex symbols stripped out."""
        trans = str.maketrans('', '', '$\\')
        return {k: v.translate(trans) for k, v in self.labels.items()}

    def xy_grids(self):
        """Expand x and y into 2D arrays matching data."""
        xgrid = np.column_stack([self.x] * self.y.size)
        ygrid = np.row_stack([self.y] * self.x.size)
        return xgrid, ygrid

    def save_txt(self, filename):
        """Save text file with 3 columns: x, y, data.

        Parameters
        ----------
        filename : str
        """
        with open(filename, 'w') as file:
            file.write("#{x:>11} {y:>12} {data:>12}\n".format(**self.plain_labels))

            xgrid, ygrid = self.xy_grids()
            for row in zip(xgrid.flat, ygrid.flat, self.data.flat):
                values = ("{:12.5e}".format(v) for v in row)
                file.write(" ".join(values) + "\n")

    def filter(self, idx_x=None, idx_y=None):
        idx_x = np.ones(self.x.size, dtype=bool) if idx_x is None else idx_x
        idx_y = np.ones(self.y.size, dtype=bool) if idx_y is None else idx_y

        self.x = self.x[idx_x]
        self.y = self.y[idx_y]
        self.data = self.data[np.ix_(idx_x, idx_y)]

    def crop(self, x=None, y=None):
        """Crop data to limits in the x and/or y axes.

        A call with x=(-1, 2) will leave data only where -1 < x < 2.

        Parameters
        ----------
        x, y : Tuple[float, float]
            Min and max data limit.
        """
        xlim, ylim = x, y
        idx_x = np.logical_and(self.x >= xlim[0], self.x <= xlim[1]) if xlim else None
        idx_y = np.logical_and(self.y >= ylim[0], self.y <= ylim[1]) if ylim else None
        self.filter(idx_x, idx_y)

    def mirror(self, axis='x'):
        """Mirror data in the specified axis. Only makes sense if the axis starts at 0.

        Parameters
        ----------
        axis : 'x' or 'y'
        """
        if axis == 'x':
            self.x = np.concatenate((-self.x[::-1], self.x[1:]))
            self.data = np.vstack((self.data[::-1], self.data[1:]))
        elif axis == 'y':
            self.y = np.concatenate((-self.y[::-1], self.y[1:]))
            self.data = np.hstack((self.data[:, ::-1], self.data[:, 1:]))

    def interpolate(self, multiply=None, size=None, kind='linear'):
        """Interpolate data using scipy's interp1d.

        Call with multiply=2 to double the size of the x-axis and interpolate data to match.
        To interpolate in both axes pass a tuple, e.g. multiply=(4, 2).

        Parameters
        ----------
        multiply : int or tuple of int
            Number of times the size of the axes should be multiplied.
        size : int or tuple of int
            New size of the axes. Zero will leave size unchanged.
        kind
            Passed to scipy.interpolate.interp1d.
        """
        from scipy.interpolate import interp1d
        if not multiply and not size:
            return

        if multiply:
            mul_x, mul_y = multiply if isinstance(multiply, tuple) else (multiply, 1)
            size_x = self.x.size * mul_x
            size_y = self.y.size * mul_y
        else:
            size_x, size_y = size if isinstance(size, tuple) else (size, 0)

        if size_x > 0 and size_x != self.x.size:
            interp_x = interp1d(self.x, self.data, axis=0, kind=kind)
            self.x = np.linspace(self.x.min(), self.x.max(), size_x, dtype=self.x.dtype)
            self.data = interp_x(self.x)

        if size_y > 0 and size_y != self.y.size:
            interp_y = interp1d(self.y, self.data, kind=kind)
            self.y = np.linspace(self.y.min(), self.y.max(), size_y, dtype=self.x.dtype)
            self.data = interp_y(self.y)

    def convolve_gaussian(self, sigma, axis='x'):
        """Convolve the data with a Gaussian function.

        Parameters
        ----------
        sigma : float
            Gaussian broadening.
        axis : 'x' or 'y'
        """
        def convolve(v, data0):
            v0 = v[v.size // 2]
            gaussian = np.exp(-0.5 * ((v - v0) / sigma)**2)
            gaussian /= gaussian.sum()

            extend = 10  # TODO: rethink this
            data = np.concatenate((data0[extend::-1], data0, data0[:-extend:-1]))
            data = np.convolve(data, gaussian, 'same')
            return data[extend:-extend]

        if 'x' in axis:
            for i in range(self.y.size):
                self.data[:, i] = convolve(self.x, self.data[:, i])

        if 'y' in axis:
            for i in range(self.x.size):
                self.data[i, :] = convolve(self.y, self.data[i, :])

    def slice_x(self, x):
        """Return a slice of data nearest to x and the found values of x.

        Parameters
        ----------
        x : float
        """
        idx = np.abs(self.x - x).argmin()
        return self.data[idx, :], self.x[idx]

    def slice_y(self, y):
        """Return a slice of data nearest to y and the found values of y.

        Parameters
        ----------
        y : float
        """
        idx = np.abs(self.y - y).argmin()
        return self.data[:, idx], self.y[idx]

    def plot(self, cbar_props=None, **kwargs):
        kwargs = with_defaults(kwargs, cmap='RdYlBu_r', rasterized=True)
        mesh = plt.pcolormesh(self.x, self.y, self.data.T, **kwargs)
        plt.xlim(self.x.min(), self.x.max())
        plt.ylim(self.y.min(), self.y.max())

        if cbar_props is not False:
            pltutils.colorbar(label=self.labels['data'])

        plt.title(self.labels['title'])
        plt.xlabel(self.labels['x'])
        plt.ylabel(self.labels['y'])

        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()

        return mesh

    def _plot_slice(self, axis, x, y, value, **kwargs):
        plt.plot(x, y, **kwargs)

        split = self.labels[axis].split(' ', 1)
        label = split[0]
        unit = '' if len(split) == 1 else split[1].strip('()')
        plt.title('{}, {} = {:.2g} {}'.format(self.labels['title'], label, value, unit))

        plt.xlim(x.min(), x.max())
        plt.xlabel(self.labels['x' if axis == 'y' else 'y'])
        plt.ylabel(self.labels['data'])
        pltutils.despine()

    def plot_slice_x(self, x, **kwargs):
        z, value = self.slice_x(x)
        self._plot_slice('x', self.y, z, value, **kwargs)

    def plot_slice_y(self, y, **kwargs):
        z, value = self.slice_y(y)
        self._plot_slice('y', self.x, z, value, **kwargs)


@pickleable
class NDSweep:
    """ND parameter sweep.

    Attributes
    ----------
    variables : tuple of array_like
        The parameters being swept.
    data : np.ndarray
        Main result array with `shape == [len(v) for v in variables]`.
    labels : dict
        Plot labels: 'title', 'x', 'y' and 'data'.
    tags : dict
        Any additional user defined variables.
    """
    def __init__(self, variables, data, labels=None, tags=None):
        self.variables = variables
        self.data = np.reshape(data, [len(v) for v in variables])

        self.labels = with_defaults(labels, title="", axes=[], data="data")
        # alias the first 3 axes to x, y, z for compatibility with Sweep labels
        for axis, label in zip('xyz', self.labels['axes']):
            self.labels[axis] = label

        self.tags = tags

    def copy(self) -> 'NDSweep':
        return deepcopy(self)
