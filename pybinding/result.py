import _pybinding
from .plot import utils as pltutils
from .utils import with_defaults
import matplotlib.pyplot as plt
import numpy as np


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


class DOS(_pybinding.DOS):
    def plot(self, **kwargs):
        plt.plot(self.energy, self.dos, **kwargs)
        plt.xlim(self.energy.min(), self.energy.max())
        plt.ylabel('DOS')
        plt.xlabel('E (eV)')
        pltutils.despine()


class LDOSenergy(_pybinding.LDOSenergy):
    def plot(self, limits=None, grid=(250, 250), **kwargs):
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        x, y = self.system.x, self.system.y

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
        grid_z = griddata((x, y), self.ldos, (grid_x, grid_y), method='cubic')

        mesh = plt.pcolormesh(grid_x, grid_y, grid_z, **with_defaults(kwargs, cmap='YlGnBu'))
        pltutils.colorbar(mesh, pad=0.02, aspect=28)

        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])


def dos(model, energy: np.ndarray, broadening: float):
    return model.calculate(DOS(energy, broadening))


def ldos_energy(model, energy: float, broadening: float, sublattice=-1):
    res = model.calculate(LDOSenergy(energy, broadening, sublattice))
    res.system = model.system
    return res
