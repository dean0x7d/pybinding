import _pybinding
from . import shape as _shape
from .support.sparse import SparseMatrix as _SparseMatrix

import matplotlib.pyplot as _plt
import numpy as _np


class System(_pybinding.System):
    @property
    def matrix(self) -> _SparseMatrix:
        matrix = self._matrix
        matrix.__class__ = _SparseMatrix
        return matrix

    @property
    def positions(self):
        return self.x, self.y, self.z

    @staticmethod
    def _plot_hoppings(ax, positions, hoppings, width, offset=(0, 0, 0), boundary=False, **kwargs):
        if width == 0:
            return

        ndims = 3 if ax.name == '3d' else 2
        offset = offset[:ndims]
        positions = positions[:ndims]

        if not boundary:
            # positions += offset
            positions = tuple(v + v0 for v, v0 in zip(positions, offset))
            # coor = x[n], y[n], z[n]
            coor = lambda n: tuple(v[n] for v in positions)
            lines = ((coor(i), coor(j)) for i, j in hoppings.indices())
        else:
            coor = lambda n: tuple(v[n] for v in positions)
            coor_plus = lambda n: tuple(v[n] + v0 for v, v0 in zip(positions, offset))
            coor_minus = lambda n: tuple(v[n] - v0 for v, v0 in zip(positions, offset))

            from itertools import chain
            lines = chain(
                ((coor_plus(i), coor(j)) for i, j in hoppings.indices()),
                ((coor(i), coor_minus(j)) for i, j in hoppings.indices())
            )

        if ndims == 2:
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(lines, lw=width, **kwargs))
            ax.autoscale_view()
        else:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            had_data = ax.has_data()
            ax.add_collection3d(Line3DCollection(list(lines), lw=width, **kwargs))

            ax.set_zmargin(0.5)
            minmax = tuple((v.min(), v.max()) for v in positions)
            ax.auto_scale_xyz(*minmax, had_data=had_data)

    @staticmethod
    def _plot_sites(ax, positions, sublattice, radius, offset=(0, 0, 0), **kwargs):
        if radius == 0:
            return

        # create array of (x, y) points
        points = _np.column_stack(v + v0 for v, v0 in zip(positions[:2], offset[:2]))

        if ax.name != '3d':
            from pybinding.support.collections import CircleCollection
            col = CircleCollection(radius, offsets=points, transOffset=ax.transData, **kwargs)
            col.set_array(sublattice)

            ax.add_collection(col)
            ax.autoscale_view()
        else:
            from pybinding.support.collections import Circle3DCollection
            col = Circle3DCollection(radius/8, offsets=points, transOffset=ax.transData, **kwargs)
            col.set_array(sublattice)
            z = positions[2] + offset[2]
            col.set_3d_properties(z, 'z')

            had_data = ax.has_data()
            ax.add_collection(col)
            minmax = tuple((v.min(), v.max()) for v in positions)
            ax.auto_scale_xyz(*minmax, had_data=had_data)

    def plot(self, colors: list=None, site_radius=0.025, site_props: dict=None,
             hopping_width=1, hopping_props: dict=None):
        """
        Parameters
        ----------
        colors : list
            list of colors to use for the different sublattices
        site_radius : float
            radius [data units] of the circle prepresenting a lattice site
        site_props : `~matplotlib.collections.Collection` properties
            additional plot options for sites
        hopping_width : float
            width [figure units] of the hopping lines
        hopping_props : `~matplotlib.collections.Collection` properties
            additional plot options for hoppings
        """
        ax = _plt.gca()
        ax.set_aspect('equal')
        ax.set_xmargin(0.01)
        ax.set_ymargin(0.01)

        # position, sublattice and hopping
        pos = self.x, self.y, self.z
        sub = self.sublattice
        hop = self.matrix

        # plot hopping lines between sites
        hopping_defaults = dict(alpha=0.5, color='black', zorder=-1)
        hopping_props = dict(hopping_defaults, **(hopping_props if hopping_props else {}))
        self._plot_hoppings(ax, pos, hop, hopping_width, **hopping_props)

        # plot site positions
        site_defaults = dict(alpha=0.85, lw=0.1)
        site_props = dict(site_defaults, **(site_props if site_props else {}))
        if colors:  # colormap with an integer norm to match the sublattice indices
            from matplotlib.colors import ListedColormap, BoundaryNorm
            site_props['cmap'] = ListedColormap(colors)
            site_props['norm'] = BoundaryNorm(list(range(len(colors)+1)), len(colors))
        self._plot_sites(ax, pos, sub, site_radius, **site_props)

        # plot periodic part
        for boundary in self.boundaries:
            shift = boundary.shift

            # shift the main sites and hoppings with lowered alpha
            kwargs = dict(site_props, alpha=site_props['alpha'] * 0.5)
            self._plot_sites(ax, pos, sub, site_radius, shift, **kwargs)
            self._plot_sites(ax, pos, sub, site_radius, -shift, **kwargs)

            kwargs = dict(hopping_props, alpha=hopping_props['alpha'] * 0.5)
            self._plot_hoppings(ax, pos, hop, hopping_width, shift, **kwargs)
            self._plot_hoppings(ax, pos, hop, hopping_width, -shift, **kwargs)

            # special color for the boundary hoppings
            b_hop = boundary.matrix
            b_hop.__class__ = _SparseMatrix
            kwargs = dict(hopping_props, color='red')
            self._plot_hoppings(ax, pos, b_hop, hopping_width, shift, boundary=True, **kwargs)

        _plt.xlabel("x (nm)")
        _plt.ylabel("y (nm)")

    def _debug_plot(self, indices=False, shape=False):
        # show the Hamiltonian index next to each atom (for debugging)
        if indices:
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                _plt.annotate(
                    str(i), (x, y), xycoords='data', color='black',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.5, lw=0.5)
                )

        if shape:
            kwargs = dict(color='black')
            if isinstance(self.shape, _shape.Polygon):
                # a polygon is just a collection of points
                _plt.plot(_np.append(self.shape.x, self.shape.x[0]),
                          _np.append(self.shape.y, self.shape.y[0]),
                          **kwargs)
            elif isinstance(self.shape, _pybinding.Circle):
                # special handling for a circle
                center = (self.shape.center[0], self.shape.center[1])
                circle = _plt.Circle(center, self.shape.r, fill=False, **kwargs)
                _plt.gca().add_artist(circle)
