import _pybinding
from .support.sparse import SparseMatrix as _SparseMatrix


class System(_pybinding.System):
    @property
    def matrix(self) -> _SparseMatrix:
        matrix = self._matrix
        matrix.__class__ = _SparseMatrix
        return matrix

    @property
    def positions(self):
        return self.x, self.y, self.z

    def plot(self, colors: list=None, site_radius=0.025, site_props: dict=None,
             hopping_width=1, hopping_props: dict=None, boundary_color='red'):
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
        import matplotlib.pyplot as plt
        from pybinding.plot.system import plot_sites, plot_hoppings

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xmargin(0.01)
        ax.set_ymargin(0.01)

        # position, sublattice and hopping
        pos = self.x, self.y, self.z
        sub = self.sublattice
        hop = self.matrix
        site_props = site_props if site_props else {}
        hopping_props = hopping_props if hopping_props else {}

        # plot main part
        plot_hoppings(ax, pos, hop, hopping_width, **hopping_props)
        plot_sites(ax, pos, sub, site_radius, colors, **site_props)

        # plot periodic part
        for boundary in self.boundaries:
            shift = boundary.shift

            # shift the main sites and hoppings with lowered alpha
            kwargs = dict(site_props, alpha=0.5)
            plot_sites(ax, pos, sub, site_radius, colors, shift, **kwargs)
            plot_sites(ax, pos, sub, site_radius, colors, -shift, **kwargs)

            kwargs = dict(hopping_props, alpha=0.25)
            plot_hoppings(ax, pos, hop, hopping_width, shift, **kwargs)
            plot_hoppings(ax, pos, hop, hopping_width, -shift, **kwargs)

            # special color for the boundary hoppings
            from pybinding.support.sparse import SparseMatrix
            b_hop = boundary.matrix
            b_hop.__class__ = SparseMatrix
            kwargs = dict(hopping_props, color=boundary_color)
            plot_hoppings(ax, pos, b_hop, hopping_width, shift, boundary=True, **kwargs)

        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")

        def clamp_to_min_limit(lim_func, min_limit=0.35):
            vmin, vmax = lim_func()
            if abs(vmax - vmin) < 2 * min_limit:
                v = (vmax + vmin) / 2
                vmin, vmax = v - min_limit, v + min_limit
                lim_func(vmin, vmax)

        clamp_to_min_limit(plt.xlim)
        clamp_to_min_limit(plt.ylim)
