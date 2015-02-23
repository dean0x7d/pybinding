import _pybinding
import numpy as _np
import matplotlib.pyplot as _plt


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

    @staticmethod
    def reduce_degenerate_wavefunctions(energy, psi, idx, epsilon=1e-5):
        # look for degenerate states
        indices = _np.argwhere(_np.abs(energy[idx] - energy) < epsilon).flat
        if len(indices) > 1:
            print('Degenerate states: {}-{} ({})'.format(indices[0], indices[-1], len(indices)))

        return _np.sum(abs(psi[indices, :])**2, axis=0).squeeze()

    def _draw_bonds(self, indices=None, alpha=0.5):
        if indices is None:
            indices = range(len(self.system.x))
        x, y = map(lambda v: v[indices], (self.system.x, self.system.y))
        bonds = self.system.matrix.to_scipy_csr()[indices]

        a, b = [], []
        for i, j in zip(*bonds.nonzero()):
            a.extend([x[i], self.system.x[j]])
            b.extend([y[i], self.system.y[j]])

        # create point pairs that define the lines
        lines = (((x1, y1), (x2, y2)) for x1, y1, x2, y2
                 in zip(a[0::2], b[0::2], a[1::2], b[1::2]))

        from matplotlib.collections import LineCollection
        _plt.gca().add_collection(LineCollection(lines, color='black', alpha=0.5*alpha, zorder=-1))

    def _plot_psi(self, index, reduce_degenerate=True):
        if reduce_degenerate:
            p = self.reduce_degenerate_wavefunctions(self.energy, self.psi, index)
        else:
            p = _np.abs(self.psi[index]).astype(_np.float32)**2

        _plt.gca().set_aspect('equal')
        _plt.xlabel('x (nm)')
        _plt.ylabel('y (nm)')

        return p

    def plot_psi_scatter(self, index, size=(20, 50), reduce_degenerate=True, limit_nm=None,
                         draw_bonds=False, color=None, **kwargs):
        x, y, sub = self.system.x, self.system.y, self.system.sublattice
        p = self._plot_psi(index, reduce_degenerate)

        if limit_nm:
            args = (_np.abs(x) < limit_nm) & (_np.abs(y) < limit_nm)
            x, y, p, sub = map(lambda v: v[args], (x, y, p, sub))
        else:
            args = range(len(x))

        if draw_bonds:
            self._draw_bonds(args)

        # sort from lowest to highest
        sort_index = _np.argsort(p)
        x, y, p, sub = map(lambda v: v[sort_index], (x, y, p, sub))

        defaults = dict(s=size[1] * p / p.max() + size[0], c=p,
                        lw=0.1, alpha=0.9, cmap='YlGnBu')
        if color:
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(color)
            bounds = list(range(len(color)+1))
            norm = BoundaryNorm(bounds, cmap.N)

            defaults.update(c=sub, cmap=cmap, norm=norm)

        sc = _plt.scatter(x, y, **dict(defaults, **kwargs))
        if not color:
            cbar = _plt.colorbar(sc, pad=0.015, aspect=28)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
        if limit_nm:
            _plt.xlim(-limit_nm, limit_nm)
            _plt.ylim(-limit_nm, limit_nm)

    def plot_psi_map(self, index, reduce_degenerate=True, limit_nm=None, grid_points=250, **kwargs):
        x, y = self.system.x, self.system.y
        p = self._plot_psi(index, reduce_degenerate)

        if limit_nm:
            from pybinding.utils import unpack_limits
            xlim, ylim = unpack_limits(limit_nm)
        else:
            xlim = x.min(), x.max()
            ylim = y.min(), y.max()

        from scipy.interpolate import griddata
        grid_x, grid_y = _np.meshgrid(
            _np.linspace(*xlim, num=grid_points),
            _np.linspace(*ylim, num=grid_points)
        )
        grid_z = griddata((x, y), p, (grid_x, grid_y), method='cubic')

        defaults = dict(cmap='YlGnBu')
        sc = _plt.pcolormesh(
            grid_x, grid_y, grid_z,
            **dict(defaults, **kwargs)
        )
        cbar = _plt.colorbar(sc, pad=0.015, aspect=28)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        _plt.xlim(*xlim)
        _plt.ylim(*ylim)
