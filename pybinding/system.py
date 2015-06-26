from collections import namedtuple

import numpy as np

import _pybinding
from .support.sparse import SparseMatrix
from .support.pickle import pickleable

Positions = namedtuple('Positions', 'x y z')


@pickleable(impl='shift matrix.')
class Boundary:
    def __init__(self, impl: _pybinding.Boundary):
        self.impl = impl

    @property
    def shift(self) -> np.ndarray:
        return self.impl.shift

    @property
    def matrix(self) -> SparseMatrix:
        return SparseMatrix(self.impl.matrix)


@pickleable(impl='positions sublattice matrix. boundaries[]')
class System:
    def __init__(self, impl: _pybinding.System):
        self.impl = impl

    @property
    def num_sites(self) -> int:
        return self.impl.num_sites

    @property
    def matrix(self) -> SparseMatrix:
        return SparseMatrix(self.impl.matrix)

    @property
    def x(self) -> np.ndarray:
        return self.impl.positions.x

    @property
    def y(self) -> np.ndarray:
        return self.impl.positions.y

    @property
    def z(self) -> np.ndarray:
        return self.impl.positions.z

    @property
    def positions(self):
        return Positions(self.x, self.y, self.z)

    @property
    def sublattice(self) -> np.ndarray:
        return self.impl.sublattice

    @property
    def boundaries(self):
        return [Boundary(b) for b in self.impl.boundaries]

    def find_nearest(self, position, sublattice=-1) -> int:
        """Find the index of the atom closest to the given coordiantes."""
        return self.impl.find_nearest(position, sublattice)

    def plot(self, site_radius: float=0.025, site_props: dict=None, hopping_width: float=1,
             hopping_props: dict=None, boundary_color: str='#ff4444', rotate: tuple=(0, 1, 2)):
        """
        Parameters
        ----------
        site_radius : float
            radius [data units] of the circle prepresenting a lattice site
        site_props : `~matplotlib.collections.Collection` properties
            additional plot options for sites
        hopping_width : float
            width [figure units] of the hopping lines
        hopping_props : `~matplotlib.collections.Collection` properties
            additional plot options for hoppings
        rotate : tuple
            axes to direction mapping:
            (0, 1, 2) -> (x, y, z) plots xy-plane
            (1, 2, 0) -> (y, z, x) plots yz-plane
        """
        import matplotlib.pyplot as plt
        import pybinding.plot.utils as pltutils
        from pybinding.plot.system import plot_sites, plot_hoppings

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

        # position, sublattice and hopping
        pos = self.x, self.y, self.z
        pos = tuple(pos[i] for i in rotate)
        sub = self.sublattice
        hop = self.matrix.tocoo()
        site_props = site_props if site_props else {}
        hopping_props = hopping_props if hopping_props else {}

        # plot main part
        plot_hoppings(ax, pos, hop, hopping_width, **hopping_props)
        plot_sites(ax, pos, sub, site_radius, **site_props)

        # plot periodic part
        for boundary in self.boundaries:
            # shift the main sites and hoppings with lowered alpha
            for shift in [boundary.shift, -boundary.shift]:
                plot_sites(ax, pos, sub, site_radius, shift, blend=0.5, **site_props)
                plot_hoppings(ax, pos, hop, hopping_width, shift, blend=0.5, **hopping_props)

            # special color for the boundary hoppings
            b_hop = boundary.matrix.tocoo()
            kwargs = dict(hopping_props, colors=boundary_color) if boundary_color else hopping_props
            plot_hoppings(ax, pos, b_hop, hopping_width, boundary.shift, boundary=True, **kwargs)

        pltutils.set_min_range(0.5)
        pltutils.despine(trim=True)
        pltutils.add_margin()
