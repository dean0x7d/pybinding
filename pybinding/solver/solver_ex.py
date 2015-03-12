import _pybinding
import numpy as _np
import matplotlib.pyplot as _plt
import pybinding.plot.utils as pltutils
from pybinding.utils import with_defaults


class SolverEx(_pybinding.Solver):
    def __init__(self):
        super().__init__()
        self.system = None

    @property
    def psi(self):
        """
        @return: Wavefunctions
        @rtype: ndarray
        """
        # transpose because it's easier to access the state number as the first index
        return super().psi.transpose()

    def save(self, file):
        _np.savez_compressed(file, energy=self.energy, psi=self.psi)

    @staticmethod
    def find_degenerate_states(energy, epsilon=1e-5):
        """Returns groups of indices which belong to the degenerate states."""
        degenerate_states = set()
        for e in energy:
            index_group = _np.argwhere(abs(e - energy) < epsilon).flat
            if len(index_group) > 1:
                degenerate_states.add(tuple(index_group))

        return degenerate_states

    def _reduce_degenerate_energy(self, pos):
        # intensity of wavefunction^2 at the given position for every state
        atom_idx = self.system.find_nearest(pos)
        p0 = abs(self.psi[:, atom_idx])**2
        p = p0.copy()

        # the instensity of each degenerate state is updated to: sum_N / N
        degenerate_states = self.find_degenerate_states(self.energy)
        for indices in degenerate_states:
            indices = list(indices)  # convert tuple to list for 1D ndarray indexing
            p[indices] = _np.sum(p0[indices]) / len(indices)

        return p, degenerate_states

    def _plot_eigenvalues_common(self, degenerate_states, show_numbers):
        """Common elements for the two eigenvalue plots."""
        if degenerate_states is not None:
            # draw lines between degenerate states
            from matplotlib.collections import LineCollection
            lines = []
            for indices in degenerate_states:
                i, j = indices[0], indices[-1]
                lines.append(((i, self.energy[i]), (j, self.energy[j])))
            _plt.gca().add_collection(LineCollection(lines, color='black', alpha=0.8))

        if show_numbers:
            # draw a number next to each state
            for i in range(len(self.energy)):
                _plt.annotate(
                    '{}'.format(i), (i, self.energy[i]), xycoords='data',
                    xytext=(0, -10), textcoords='offset points', fontsize='x-small',
                    horizontalalignment='center', color='black', bbox=None
                )

        _plt.xlabel('state')
        _plt.ylabel('E (eV)')
        _plt.xlim(-1, len(self.energy))
        locs, _ = _plt.xticks()
        _plt.xticks([x for x in locs if 0 <= x < len(self.energy)])

    def plot_eigenvalues(self, mark_degenerate=True, show_numbers=False, **kwargs):
        """Standard eigenvalues scatter plot."""
        state_numbers = _np.arange(0, len(self.energy))
        defaults = dict(c='blue', s=15, lw=0.2)
        _plt.scatter(state_numbers, self.energy, **dict(defaults, **kwargs))

        degenerate_states = self.find_degenerate_states(self.energy) if mark_degenerate else None
        self._plot_eigenvalues_common(degenerate_states, show_numbers)

    def plot_eigenvalues_cmap(self, pos=None, size=(7, 77), mark_degenerate=False,
                              show_numbers=False, label_max=False, sort=True, **kwargs):
        """Eigenvalues scatter plot with a colormap.

        The colormap indicates wavefunction intensity at the given position.
        """

        energy = self.energy
        state_numbers = _np.arange(0, len(energy))
        intensity, degenerate_states = self._reduce_degenerate_energy(pos)
        max_index = intensity.argmax()

        if sort:
            # sort from lowest to highest
            sort_index = _np.argsort(intensity)
            state_numbers, energy, intensity = map(lambda v: v[sort_index],
                                                   (state_numbers, energy, intensity))

        defaults = dict(c=intensity, s=size[1] * intensity / intensity.max() + size[0],
                        cmap='YlOrRd', lw=0.2, alpha=0.85)
        _plt.scatter(state_numbers, energy, **dict(defaults, **kwargs))
        cbar = _plt.colorbar(pad=0.015, aspect=28)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        # indicate the position of max intensity
        if label_max:
            indices = next((i for i in degenerate_states if max_index in i), None)
            if indices:
                show_index = (indices[0] + indices[-1]) / 2
            else:
                show_index = max_index

            _plt.annotate(
                "{}: {:.3f} eV".format(max_index, self.energy[max_index]),
                xy=(show_index, self.energy[max_index]), xycoords='data',
                xytext=(0, 8), textcoords='offset points', horizontalalignment='center'
            )

        self._plot_eigenvalues_common(degenerate_states if mark_degenerate else None, show_numbers)
        return max_index

    def get_degenerate_indices(self, index, epsilon=1e-5):
        """Return a lies of degenerate energy indices (within tolerange epsilon)"""
        return _np.argwhere(_np.abs(self.energy[index] - self.energy) < epsilon)[:, 0]

    def get_intensity(self, indices):
        """Return sum of wavefunction^2 at indices"""
        return _np.sum(abs(self.psi[indices, :])**2, axis=0).squeeze()

    def plot_wavefunction(self, index, reduce=1e-5, site_radius=(0.03, 0.05), hopping_width=1,
                          hopping_props=None, cbar_props=None, **kwargs):
        from pybinding.plot.system import plot_hoppings, plot_sites

        ax = _plt.gca()
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        if reduce:
            index = self.get_degenerate_indices(index, reduce)
        intensity = self.get_intensity(index)

        radius = site_radius[0] + site_radius[1] * intensity / intensity.max()
        collection = plot_sites(ax, self.system.positions, intensity, radius,
                                **with_defaults(kwargs, cmap='YlGnBu'))

        plot_hoppings(ax, self.system.positions, self.system.matrix, hopping_width,
                      **with_defaults(hopping_props, colors='#bbbbbb'))

        pltutils.colorbar(collection, **with_defaults(cbar_props, pad=0.015, aspect=28))
        pltutils.despine(trim=True)
        pltutils.add_margin()

        return index

    def plot_wavefunction_mesh(self, index, reduce=1e-5, limits=None, grid=(250, 250), **kwargs):
        ax = _plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        x, y = self.system.x, self.system.y
        if reduce:
            index = self.get_degenerate_indices(index, reduce)
        intensity = self.get_intensity(index)

        if not limits:
            limits = x.min(), x.max(), y.min(), y.max()
        xlim, ylim = limits[:2], limits[2:]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        from scipy.interpolate import griddata
        grid_x, grid_y = _np.meshgrid(
            _np.linspace(*xlim, num=grid[0]),
            _np.linspace(*ylim, num=grid[1])
        )
        grid_z = griddata((x, y), intensity, (grid_x, grid_y), method='cubic')

        mesh = _plt.pcolormesh(grid_x, grid_y, grid_z, **with_defaults(kwargs, cmap='YlGnBu'))
        pltutils.colorbar(mesh, pad=0.02, aspect=28)

        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])

        return index
