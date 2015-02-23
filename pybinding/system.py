import _pybinding
from . import shape as _shape
from .support.sparse import SparseMatrix as _SparseMatrix

import matplotlib.pyplot as _plt
import numpy as _np


class System(_pybinding.System):
    plot3d = False

    @property
    def matrix(self) -> _SparseMatrix:
        matrix = self._matrix
        matrix.__class__ = _SparseMatrix
        return matrix

    def draw_bonds(self, ax, offset=(0, 0, 0), alpha=1.0):
        x0, y0, z0 = offset
        x, y, z = [], [], []

        for i, j in self.matrix.indices():
            x.extend([self.x[i] + x0, self.x[j] + x0])
            y.extend([self.y[i] + y0, self.y[j] + y0])
            z.extend([self.z[i] + z0, self.z[j] + z0])

        if not self.plot3d:
            # create point pairs that define the lines
            lines = (((x1, y1), (x2, y2)) for x1, y1, x2, y2
                     in zip(x[0::2], y[0::2], x[1::2], y[1::2]))

            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(lines, color='black', alpha=0.5*alpha, zorder=-1))
        else:
            # create point pairs that define the lines
            lines = [((x1, y1, z1), (x2, y2, z2)) for x1, y1, z1, x2, y2, z2
                     in zip(x[0::2], y[0::2], z[0::2], x[1::2], y[1::2], z[1::2])]

            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            ax.add_collection3d(Line3DCollection(lines, color='black', alpha=0.5*alpha, zorder=-1))

    def plot(self, color=None, draw_bonds=True, shape_border='', draw_numbers=False, plot3d=False,
             **kwargs):
        """
        Parameters
        ----------
        color : list
            list of colors to use for the different sublattices
        s : int
            dot size
        alpha : float
            dot alpha transparency
        draw_bonds : bool
            bond lines between atom (may be slow for big systems)
        shape_border : str
            color of the shape outline e.g. shape_border='red'
        draw_numbers : bool
            for debug only

        Other parameters
        ----------------
        kwargs : `~matplotlib.collections.Collection` properties
            anything that matplotlib scatter takes
        """

        self.plot3d = plot3d
        if not plot3d:
            ax = _plt.gca()
            z = ()
        else:
            from mpl_toolkits.mplot3d import Axes3D
            ax = Axes3D(_plt.gcf())
            z = (self.z, )
        ax.set_aspect('equal')

        # draw bond lines between atoms
        if draw_bonds:
            self.draw_bonds(ax)

        # set default plot arguments if needed
        defaults = dict(c=self.sublattice, s=20, alpha=0.8, lw=0.8)
        kwargs = dict(defaults, **kwargs)

        # sublattice colors
        if color is not None:
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(color)
            bounds = list(range(len(color)+1))
            norm = BoundaryNorm(bounds, cmap.N)

            kwargs['cmap'] = cmap
            kwargs['norm'] = norm

        # plot the atom positions
        ax.scatter(self.x, self.y, *z, **kwargs)

        # draw periodic part
        for periodic in self.boundaries:
            shift = periodic.shift
            matrix = periodic.matrix
            matrix.__class__ = _SparseMatrix

            kwargs['alpha'] = 0.4
            ax.scatter(self.x + shift[0], self.y + shift[1], *z, **kwargs)
            ax.scatter(self.x - shift[0], self.y - shift[1], *z, **kwargs)
            self.draw_bonds(ax, offset=shift, alpha=0.4)
            self.draw_bonds(ax, offset=-shift, alpha=0.4)

            for i, j in matrix.indices():
                zs = ()
                if plot3d:
                    zs = [self.z[i], self.z[j]],

                ax.plot([self.x[i]+shift[0], self.x[j]],
                        [self.y[i]+shift[1], self.y[j]],
                        *zs,
                        color='red', alpha=0.5, zorder=-1)
                ax.plot([self.x[i], self.x[j]-shift[0]],
                        [self.y[i], self.y[j]-shift[1]],
                        *zs,
                        color='red', alpha=0.5, zorder=-1)

        # draw the Hamiltonian index next to each atom (for debugging)
        if draw_numbers:
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                ax.annotate("{}".format(i), (x, y), xycoords='data',
                            xytext=(2, 5), textcoords='offset points',
                            )

        # draw the shape outline
        if shape_border != '':
            if isinstance(self.shape, _shape.Polygon):
                # a polygon is just a collection of points
                ax.plot(_np.append(self.shape.x, self.shape.x[0]),
                        _np.append(self.shape.y, self.shape.y[0]),
                        color=shape_border)
            elif isinstance(self.shape, _shape.Circle):
                # special handling for a circle
                center = (self.shape.center[0], self.shape.center[1])
                circle = _plt.Circle(center, self.shape.r, color=shape_border, fill=False)
                ax.add_artist(circle)

        _plt.xlabel("x (nm)")
        _plt.ylabel("y (nm)")
