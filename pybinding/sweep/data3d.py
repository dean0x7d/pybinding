from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .. import pltutils
from ..utils import with_defaults


class Data3D:
    def __init__(self, file_name=None, title='', description='', labels=defaultdict(str)):
        self.file_name = file_name
        self.title = title
        self.description = description
        self.labels = labels
        self.fields = ['x', 'y', 'z', 'title', 'description', 'labels']
        self.x, self.y, self.z = (np.array([]),) * 3  # just so IDEs know these are ndarrays
        self.load()

    def copy(self) -> 'Data3D':
        import copy
        return copy.copy(self)

    def update_strings(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.fields:
                self.__dict__[name] = value

    def plain_labels(self):
        """ Strip latex from label name """
        return {k: v.strip('$\\') for k, v in self.labels.items()}

    def expand_xy(self):
        x = np.repeat(self.x, self.y.size).reshape((self.x.size, self.y.size))
        y = np.tile(self.y, self.x.size).reshape((self.x.size, self.y.size))
        return x, y

    def crop(self, x=None, y=None):
        def cut(v, limits):
            if limits is not None:
                index = (v >= limits[0]) & (v <= limits[1])
                return v[index], index
            else:
                return v, range(v.size)

        self.x, x_index = cut(self.x, x)
        self.y, y_index = cut(self.y, y)
        self.z = self.z[np.ix_(x_index, y_index)]

    def mirror(self, axis='x'):
        if axis == 'x':
            self.x = np.hstack((-self.x[::-1], self.x[1:]))
            self.z = np.vstack((self.z[::-1], self.z[1:]))
        elif axis == 'y':
            raise Exception('Not implemented yet')

    def interpolate(self, multiply=None, size=None, kind='linear'):
        from scipy.interpolate import interp1d
        size_x, size_y = self.x.size, self.y.size
        if multiply is not None:
            mul_x, mul_y = multiply if isinstance(multiply, tuple) else (multiply, 1)
            size_x = self.x.size * mul_x
            size_y = self.y.size * mul_y
        else:
            size_x, size_y = size if isinstance(size, tuple) else (size, size_y)

        if size_x != self.x.size and size_x != -1:
            interp_x = interp1d(self.x, self.z, axis=0, kind=kind)
            self.x = np.linspace(self.x.min(), self.x.max(), size_x, dtype=np.float32)
            self.z = interp_x(self.x)

        if size_y != self.y.size and size_y != -1:
            interp_y = interp1d(self.y, self.z, kind=kind)
            self.y = np.linspace(self.y.min(), self.y.max(), size_y, dtype=np.float32)
            self.z = interp_y(self.y)

    def convolve_gaussian(self, sigma, extend=10):
        def convolve(x, z0):
            gaussian = np.exp(-0.5 * ((x - x[x.size / 2]) / sigma)**2)
            gaussian /= gaussian.sum()

            z = np.concatenate((z0[extend::-1], z0, z0[:-extend:-1]))
            z = np.convolve(z, gaussian, 'same')

            z = z[extend:-extend]
            return z

        for i in range(self.y.size):
            self.z[:, i] = convolve(self.x, self.z[:, i])

    def get(self):
        return self.x, self.y, self.z

    def get_for_plot(self):
        return self.x, self.y, self.z.transpose()

    def get_flat(self):
        x, y = self.expand_xy()
        return x.flatten(), y.flatten(), self.z.flatten()

    def slice(self, x=None, y=None):
        """Get a 1D slice of z closest to the given point on the x or y axis (never both)"""
        if x is not None:
            idx = np.abs(self.x - x).argmin()
            return self.z[idx, :], self.x[idx]
        elif y is not None:
            idx = np.abs(self.y - y).argmin()
            return self.z[:, idx], self.y[idx]

    def save(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        if not file_name:
            return

        kwargs = {name: self.__dict__[name] for name in self.fields}
        np.savez_compressed(file_name, **kwargs)

    def load(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        if not file_name:
            return
        data = np.load(file_name)

        # load all fields from file: self.<name> = data[<name>]
        for name in self.fields:
            if not self.__dict__[name]:
                self.__dict__[name] = data[name]

        # labels are stored as repr() of dict(), so to get the value
        import ast
        self.labels = ast.literal_eval(str(self.labels))

    def export_text(self, file_name):
        with open(file_name, 'w') as file:
            file.write('# {x:12}{y:13}{z:13}\n'.format(**self.labels))
            for x, y, z in zip(*self.get_flat()):
                file.write(("{:13.5e}"*3 + '\n').format(x, y, z))

    def plot(self, cbar_props=None, **kwargs):
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()

        mesh = plt.pcolormesh(*self.get_for_plot(), **with_defaults(kwargs, cmap='RdYlBu_r'))
        plt.xlim(self.x.min(), self.x.max())
        plt.ylim(self.y.min(), self.y.max())

        if cbar_props is not False:
            cbar = plt.colorbar(**with_defaults(cbar_props, pad=0.015, aspect=30))
            cbar.ax.set_xlabel(self.labels['z'])
            cbar.ax.xaxis.set_label_position('top')

        plt.title(self.title)
        plt.xlabel(self.labels['x'])
        plt.ylabel(self.labels['y'])

        return mesh

    def plot_slice(self, x=None, y=None, **kwargs):
        z, value = self.slice(x, y)

        v = 'y' if x is not None else 'x'
        v_array = getattr(self, v)
        plt.plot(v_array, z, **kwargs)

        split = self.labels['x' if x is not None else 'y'].split(' ', 1)
        label = split[0]
        unit = '' if len(split) == 1 else split[1].strip('()')
        plt.title('{}, {} = {:.2g} {}'.format(self.title, label, value, unit))

        plt.xlim(v_array.min(), v_array.max())
        plt.xlabel(self.labels[v])
        plt.ylabel(self.labels['z'])
        pltutils.despine()

        return value
