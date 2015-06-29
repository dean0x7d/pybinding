import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from . import pltutils
from .utils import with_defaults
from .system import plot_sites, plot_hoppings
from .support.pickle import pickleable

__all__ = ['LDOSpoint', 'SpatialMap']


@pickleable
class LDOSpoint:
    def __init__(self, energy, ldos):
        self.energy = energy
        self.ldos = ldos

    def plot(self, **kwargs):
        plt.plot(self.energy, self.ldos, **kwargs)
        plt.xlim(self.energy.min(), self.energy.max())
        plt.ylabel('LDOS')
        plt.xlabel('E (eV)')
        pltutils.despine()


class SpatialMap:
    def __init__(self, data, xyz, sublattice, hoppings: csr_matrix):
        self.data = data
        self.xyz = xyz
        self.sublattice = sublattice
        self.hoppings = hoppings

    @classmethod
    def from_system(cls, data, system):
        return cls(data, system.positions, system.sublattice, system.matrix.tocsr())

    @classmethod
    def from_file(cls, file_name):
        file = np.load(file_name)
        xyz = tuple(file[v] for v in 'xyz')
        m = tuple(file[v] for v in ['m_data', 'm_indices', 'm_indptr'])
        return cls(file['data'], xyz, file['sublattice'], csr_matrix(m, file['m_shape']))

    def copy(self) -> 'SpatialMap':
        import copy
        return copy.copy(self)

    def save(self, file_name):
        x, y, z = self.xyz
        m = self.hoppings

        np.savez_compressed(
            file_name,
            data=self.data, x=x, y=y, z=z, sublattice=self.sublattice,
            m_data=m.data, m_indices=m.indices, m_indptr=m.indptr, m_shape=m.shape
        )

    def save_txt(self, file_name):
        with open(file_name+'.dat', 'w') as file:
            file.write('# {:12}{:13}{:13}\n'.format('x(nm)', 'y(nm)', 'data'))
            for x, y, value in zip(self.xyz[0], self.xyz[1], self.data):
                file.write(("{:13.5e}"*3 + '\n').format(x, y, value))

    def filter(self, idx):
        self.data = self.data[idx]
        self.sublattice = self.sublattice[idx]
        self.xyz = map(lambda v: v[idx], self.xyz)
        self.hoppings = self.hoppings[idx][:, idx]

    def crop(self, x=None, y=None):
        xlim, ylim = x, y
        x, y, _ = self.xyz
        idx = np.ones(x.size, dtype=bool)
        if xlim:
            idx = np.logical_and(idx, x >= xlim[0], x <= xlim[1])
        if ylim:
            idx = np.logical_and(idx, y >= ylim[0], y <= ylim[1])
        self.filter(idx)

    def convolve(self, sigma=0.25):
        x, y, _ = self.xyz
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
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])

    def plot_contourf(self, num_levels=50, cbar_props=None, **kwargs):
        levels = np.linspace(self.data.min(), self.data.max(), num_levels)
        x, y, _ = self.xyz
        contour = plt.tricontourf(x, y, self.data,
                                  **with_defaults(kwargs, cmap='YlGnBu', levels=levels))

        if cbar_props is not False:
            cbar_props = cbar_props if cbar_props else {}
            pltutils.colorbar(**with_defaults(cbar_props, format='%.2f'))

        self._decorate_plot()
        return contour

    def plot_contour(self, **kwargs):
        x, y, _ = self.xyz
        contour = plt.tricontour(x, y, self.data, **kwargs)

        self._decorate_plot()
        return contour

    def plot_structure(self, site_radius=(0.03, 0.05), site_props: dict=None, hopping_width=1,
                       hopping_props: dict=None, cbar_props: dict=None):
        ax = plt.gca()
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        x, y, z = self.xyz
        radius = site_radius[0] + site_radius[1] * self.data / self.data.max()
        collection = plot_sites(ax, (x, y, z), self.data, radius,
                                **with_defaults(site_props, cmap='YlGnBu'))

        plot_hoppings(ax, (x, y, z), self.hoppings.tocoo(), hopping_width,
                      **with_defaults(hopping_props, colors='#bbbbbb'))

        pltutils.colorbar(collection, **with_defaults(cbar_props))
        pltutils.despine(trim=True)
        pltutils.add_margin()
