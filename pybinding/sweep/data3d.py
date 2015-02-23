from collections import defaultdict as _defaultdict
import numpy as _np
import matplotlib.pyplot as _plt


class Data3D():
    def __init__(self, file_name=None, title='', description='',
                 labels=_defaultdict(str)):
        self.file_name = file_name
        self.title = title
        self.description = description
        self.labels = labels
        self.fields = ['x', 'y', 'z', 'title', 'description', 'labels']
        self.x, self.y, self.z = (_np.array([]),) * 3  # just so IDEs know these are ndarrays
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
        x = _np.repeat(self.x, self.y.size).reshape((self.x.size, self.y.size))
        y = _np.tile(self.y, self.x.size).reshape((self.x.size, self.y.size))
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
        self.z = self.z[_np.ix_(x_index, y_index)]

    def mirror(self, axis='x'):
        if axis == 'x':
            self.x = _np.hstack((-self.x[::-1], self.x[1:]))
            self.z = _np.vstack((self.z[::-1], self.z[1:]))
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
            self.x = _np.linspace(self.x.min(), self.x.max(), size_x, dtype=_np.float32)
            self.z = interp_x(self.x)

        if size_y != self.y.size and size_y != -1:
            interp_y = interp1d(self.y, self.z, kind=kind)
            self.y = _np.linspace(self.y.min(), self.y.max(), size_y, dtype=_np.float32)
            self.z = interp_y(self.y)

    def get(self):
        return self.x, self.y, self.z

    def get_for_plot(self):
        return self.x, self.y, self.z.transpose()

    def get_flat(self):
        x, y = self.expand_xy()
        return x.flatten(), y.flatten(), self.z.flatten()

    def slice_x(self, x):
        idx = _np.abs(self.x - x).argmin()
        x_point = self.x[idx]
        z_slice = self.z[idx, :]
        return x_point, self.y, z_slice

    def save(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        if not file_name:
            return

        kwargs = {name: self.__dict__[name] for name in self.fields}
        _np.savez_compressed(file_name, **kwargs)

    def load(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        if not file_name:
            return
        data = _np.load(file_name)

        # load all fields from file: self.<name> = data[<name>]
        for name in self.fields:
            if not self.__dict__[name]:
                self.__dict__[name] = data[name]

        # labels are stored as repr() of dict(), so to get the value
        import ast
        self.labels = ast.literal_eval(str(self.labels))

    def plot(self, cmap='rainbow', log_norm=False, z_limits=None):
        from matplotlib.colors import LogNorm

        lw = 0.6
        _plt.gca().tick_params(width=0.5)
        _plt.gca().get_xaxis().tick_bottom()
        _plt.gca().get_yaxis().tick_left()
        for axis in ['top', 'bottom', 'left', 'right']:
            _plt.gca().spines[axis].set_linewidth(lw)

        mesh = _plt.pcolormesh(
            *self.get_for_plot(),
            cmap=cmap,
            norm=LogNorm() if log_norm else None
        )
        _plt.xlim(self.x.min(), self.x.max())
        _plt.ylim(self.y.min(), self.y.max())
        if z_limits:
            mesh.set_clim(*z_limits)
        cbar = _plt.colorbar(aspect=35, pad=0.015)
        cbar.outline.set_linewidth(lw)  # only works after set_clim()
        cbar.ax.set_xlabel(self.labels['z'])
        cbar.ax.xaxis.set_label_position('top')

        _plt.title(self.title)
        _plt.xlabel(self.labels['x'])
        _plt.ylabel(self.labels['y'])

    def plot_slice_x(self, x, **kwargs):
        x_point, y, z_slice = self.slice_x(x)
        _plt.plot(y, z_slice, **kwargs)

        x_split = self.labels['x'].split(' ', 1)
        x_label = x_split[0]
        x_unit = '' if len(x_split) == 1 else x_split[1].strip('()')
        _plt.title('{}, {} = {:.2g} {}'.format(self.title, x_label, x_point, x_unit))

        _plt.xlabel(self.labels['y'])
        _plt.ylabel(self.labels['z'])
        _plt.xlim(y.min(), y.max())

        return x_point
