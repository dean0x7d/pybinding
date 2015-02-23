import _pybinding
import matplotlib.pyplot as _plt


class LDOSpoint(_pybinding.LDOSpoint):
    def plot(self, **kwargs):
        _plt.plot(self.energy, self.ldos, **kwargs)
        _plt.xlim(self.energy.min(), self.energy.max())
        _plt.ylabel('LDOS')
        _plt.xlabel('E (eV)')


class DOS(_pybinding.DOS):
    def plot(self, **kwargs):
        _plt.plot(self.energy, self.dos, **kwargs)
        _plt.xlim(self.energy.min(), self.energy.max())
        _plt.ylabel('DOS')
        _plt.xlabel('E (eV)')


class LDOSenergy(_pybinding.LDOSenergy):
    def plot(self, limit_nm=None, grid_points=250, **kwargs):
        x, y = self.system.x, self.system.y

        if limit_nm:
            from pybinding.utils import unpack_limits
            xlim, ylim = unpack_limits(limit_nm)
        else:
            xlim = x.min(), x.max()
            ylim = y.min(), y.max()

        import numpy as _np
        grid_x, grid_y = _np.meshgrid(
            _np.linspace(*xlim, num=grid_points),
            _np.linspace(*ylim, num=grid_points)
        )
        from scipy.interpolate import griddata
        grid_z = griddata((x, y), self.ldos, (grid_x, grid_y), method='cubic')

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
        _plt.gca().set_aspect('equal')
        _plt.xlabel('x (nm)')
        _plt.ylabel('y (nm)')


def ldos_point(model, energy: 'ndarray', broadening: float, position: tuple, sublattice=-1):
    return model.calculate(LDOSpoint(energy, broadening, position, sublattice))


def dos(model, energy: 'ndarray', broadening: float):
    return model.calculate(model, DOS(energy, broadening))


def ldos_energy(model, energy: float, broadening: float, sublattice=-1):
    res = model.calculate(model, LDOSenergy(energy, broadening, sublattice))
    res.system = model.system
    return res
