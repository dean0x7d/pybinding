import _pybinding
import numpy as np
import matplotlib.pyplot as plt
from ..plot import utils as pltutils
from ..utils import with_defaults


class Solver(_pybinding.Solver):
    def __init__(self):
        super().__init__()
        self.system = None

    @property
    def psi(self) -> 'np.ndarray':
        # transpose because it's easier to access the state number as the first index
        return super().psi.transpose()

    def get_intensity(self, indices) -> 'np.ndarray':
        """Return sum of wavefunction^2 at indices"""
        return np.sum(abs(self.psi[indices, :])**2, axis=0).squeeze()

    def save(self, file):
        np.savez_compressed(file, energy=self.energy, psi=self.psi)

    @staticmethod
    def get_degenerate_indices(energy, index, epsilon=1e-5) -> 'np.ndarray':
        """Return a list of degenerate energy indices (within tolerance epsilon)"""
        return np.argwhere(np.abs(energy[index] - energy) < epsilon)[:, 0]

    @staticmethod
    def find_degenerate_states(energy, epsilon=1e-5) -> set:
        """Return groups of indices which belong to degenerate states"""
        degenerate_states = set()
        for i in range(len(energy)):
            index_group = Solver.get_degenerate_indices(energy, i, epsilon)
            if len(index_group) > 1:
                degenerate_states.add(tuple(index_group))

        return degenerate_states

    def _reduce_degenerate_energy(self, position) -> 'np.ndarray':
        # intensity of wavefunction^2 at the given position for every state
        atom_idx = self.system.find_nearest(position)
        intensity = abs(self.psi[:, atom_idx])**2
        p0 = intensity.copy()

        # the instensity of each degenerate state is updated to: sum_N / N
        states = self.find_degenerate_states(self.energy)
        for indices in states:
            indices = list(indices)  # convert tuple to list for 1D ndarray indexing
            intensity[indices] = np.sum(p0[indices]) / len(indices)

        return intensity

    def _plot_eigenvalues_common(self, mark_degenerate, number_states):
        """Common elements for the two eigenvalue plots"""
        if mark_degenerate:
            # draw lines between degenerate states
            from matplotlib.collections import LineCollection
            pairs = ((s[0], s[-1]) for s in self.find_degenerate_states(self.energy))
            lines = [[(i, self.energy[i]) for i in pair] for pair in pairs]
            plt.gca().add_collection(LineCollection(lines, color='black', alpha=0.5))

        if number_states:
            # draw a number next to each state
            for i in range(len(self.energy)):
                pltutils.annotate_box(i, (i, self.energy[i]), fontsize='x-small',
                                      xytext=(0, -10), textcoords='offset points')

        plt.xlabel('state')
        plt.ylabel('E (eV)')
        plt.xlim(-1, len(self.energy))
        xticks, _ = plt.xticks()
        plt.xticks([x for x in xticks if 0 <= x < len(self.energy)])
        pltutils.despine()

    def plot_eigenvalues(self, mark_degenerate=True, number_states=False, **kwargs):
        """Standard eigenvalues scatter plot"""
        states = np.arange(0, self.energy.size)
        plt.scatter(states, self.energy, **with_defaults(kwargs, c='#377ec8', s=15, lw=0.1))
        self._plot_eigenvalues_common(mark_degenerate, number_states)

    def plot_eigenvalues_cmap(self, position, size=(7, 77), mark_degenerate=True,
                              number_states=False, cbar_props=None, **kwargs):
        """Eigenvalues scatter plot with a colormap

        The colormap indicates wavefunction intensity at the given position.
        """
        states = np.arange(0, self.energy.size)
        intensity = self._reduce_degenerate_energy(position)
        max_index = intensity.argmax()

        # higher intensity states should be drawn above lower intensity states
        idx = np.argsort(intensity)
        states, energy, intensity = (v[idx] for v in (states, self.energy, intensity))

        kwargs = with_defaults(kwargs, cmap='YlOrRd', lw=0.2, alpha=0.85,
                               c=intensity, s=size[0] + size[1] * intensity / intensity.max())
        plt.scatter(states, energy, **kwargs)
        pltutils.colorbar(**with_defaults(cbar_props, pad=0.02, aspect=28))

        self._plot_eigenvalues_common(mark_degenerate, number_states)
        return max_index

    def plot_wavefunction(self, index, reduce=1e-5, site_radius=(0.03, 0.05), hopping_width=1,
                          hopping_props=None, cbar_props=None, limits=None, **kwargs):
        from pybinding.plot.system import plot_hoppings, plot_sites

        ax = plt.gca()
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        if reduce:
            index = self.get_degenerate_indices(self.energy, index, reduce)
        intensity = self.get_intensity(index)

        x, y, z = self.system.positions
        hoppings = self.system.matrix.tocsr()
        if limits:
            xlim, ylim = limits[:2], limits[2:]
            idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
            x, y, z, intensity = (v[idx] for v in (x, y, z, intensity))
            hoppings = hoppings[idx][:, idx]

        radius = site_radius[0] + site_radius[1] * intensity / intensity.max()
        collection = plot_sites(ax, (x, y, z), intensity, radius,
                                **with_defaults(kwargs, cmap='YlGnBu'))

        plot_hoppings(ax, (x, y, z), hoppings.tocoo(), hopping_width,
                      **with_defaults(hopping_props, colors='#bbbbbb'))

        pltutils.colorbar(collection, **with_defaults(cbar_props, pad=0.015, aspect=28))
        pltutils.despine(trim=True)
        pltutils.add_margin()

        return index

    def plot_wavefunction_mesh(self, index, reduce=1e-5, limits=None, grid=(250, 250), **kwargs):
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        x, y = self.system.x, self.system.y
        if reduce:
            index = self.get_degenerate_indices(self.energy, index, reduce)
        intensity = self.get_intensity(index)

        if not limits:
            limits = x.min(), x.max(), y.min(), y.max()
        xlim, ylim = limits[:2], limits[2:]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        from scipy.interpolate import griddata
        grid_x, grid_y = np.meshgrid(
            np.linspace(*xlim, num=grid[0]),
            np.linspace(*ylim, num=grid[1])
        )
        grid_z = griddata((x, y), intensity, (grid_x, grid_y), method='cubic')

        mesh = plt.pcolormesh(grid_x, grid_y, grid_z, **with_defaults(kwargs, cmap='YlGnBu'))
        pltutils.colorbar(mesh, pad=0.02, aspect=28)

        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])

        return index
