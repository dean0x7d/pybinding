import numpy as np
import matplotlib.pyplot as plt

import _pybinding
from .. import results
from ..system import System
from ..utils import with_defaults
from ..plot import utils as pltutils
from ..support.pickle import pickleable


@pickleable(impl='system. eigenvalues eigenvectors')
class Solver:
    def __init__(self, impl: _pybinding.Solver):
        self.impl = impl

    def set_model(self, model):
        self.impl.set_model(model)

    def solve(self):
        self.impl.solve()

    def report(self, shortform=False):
        return self.impl.report(shortform)

    @property
    def system(self) -> System:
        return System(self.impl.system)

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.impl.eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.impl.eigenvectors

    def calc_intensity(self, indices, reduce=1e-5):
        if reduce:
            indices = self.get_degenerate_indices(self.eigenvalues, indices, reduce)

        # wavefunction**2 at each index
        intensity = np.sum(abs(self.eigenvectors[:, indices])**2, axis=0).squeeze()
        return results.SpatialMap.from_system(intensity, self.system)

    def calc_dos(self, energies, broadening) -> np.ndarray:
        return self.impl.calc_dos(energies, broadening)

    def calc_ldos(self, energy, broadening, sublattice=-1):
        ldos = self.impl.calc_ldos(energy, broadening, sublattice)
        return results.SpatialMap.from_system(ldos, self.system)

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
        intensity = abs(self.eigenvectors[atom_idx, :])**2
        p0 = intensity.copy()

        # the instensity of each degenerate state is updated to: sum_N / N
        states = self.find_degenerate_states(self.eigenvalues)
        for indices in states:
            indices = list(indices)  # convert tuple to list for 1D ndarray indexing
            intensity[indices] = np.sum(p0[indices]) / len(indices)

        return intensity

    def _plot_eigenvalues_common(self, mark_degenerate, number_states):
        """Common elements for the two eigenvalue plots"""
        if mark_degenerate:
            # draw lines between degenerate states
            from matplotlib.collections import LineCollection
            pairs = ((s[0], s[-1]) for s in self.find_degenerate_states(self.eigenvalues))
            lines = [[(i, self.eigenvalues[i]) for i in pair] for pair in pairs]
            plt.gca().add_collection(LineCollection(lines, color='black', alpha=0.5))

        if number_states:
            # draw a number next to each state
            for i in range(len(self.eigenvalues)):
                pltutils.annotate_box(i, (i, self.eigenvalues[i]), fontsize='x-small',
                                      xytext=(0, -10), textcoords='offset points')

        plt.xlabel('state')
        plt.ylabel('E (eV)')
        plt.xlim(-1, len(self.eigenvalues))
        xticks, _ = plt.xticks()
        plt.xticks([x for x in xticks if 0 <= x < len(self.eigenvalues)])
        pltutils.despine()

    def plot_eigenvalues(self, mark_degenerate=True, number_states=False, **kwargs):
        """Standard eigenvalues scatter plot"""
        states = np.arange(0, self.eigenvalues.size)
        plt.scatter(states, self.eigenvalues, **with_defaults(kwargs, c='#377ec8', s=15, lw=0.1))
        self._plot_eigenvalues_common(mark_degenerate, number_states)

    def plot_eigenvalues_cmap(self, position, size=(7, 77), mark_degenerate=True,
                              number_states=False, cbar_props=None, **kwargs):
        """Eigenvalues scatter plot with a colormap

        The colormap indicates wavefunction intensity at the given position.
        """
        states = np.arange(0, self.eigenvalues.size)
        intensity = self._reduce_degenerate_energy(position)
        max_index = intensity.argmax()

        # higher intensity states should be drawn above lower intensity states
        idx = np.argsort(intensity)
        states, energy, intensity = (v[idx] for v in (states, self.eigenvalues, intensity))

        kwargs = with_defaults(kwargs, cmap='YlOrRd', lw=0.2, alpha=0.85,
                               c=intensity, s=size[0] + size[1] * intensity / intensity.max())
        plt.scatter(states, energy, **kwargs)
        pltutils.colorbar(**with_defaults(cbar_props, pad=0.02, aspect=28))

        self._plot_eigenvalues_common(mark_degenerate, number_states)
        return max_index

    def plot_bands(self, k0, k1, *ks, step, names=None):
        ks = [np.array(k) for k in (k0, k1) + ks]
        energy = []
        points = [0]
        for start, end in zip(ks[:-1], ks[1:]):
            num_steps = max(abs(end - start) / step)
            k_list = (np.linspace(s, e, num_steps) for s, e in zip(start, end))

            for k in zip(*k_list):
                self.model.set_wave_vector(k)
                energy.append(self.eigenvalues)
            points += [len(energy)-1]

        for point in points[1:-1]:
            plt.axvline(point, color='black', ls='--')
        plt.xticks(points, names if names else [])

        plt.plot(energy, color='blue')
        plt.xlim(0, len(energy) - 1)
        plt.xlabel('k-space')
        plt.ylabel('E (eV)')
