import numpy as np
from pybinding.utils import with_defaults


def blend_colors(color, bg, blend):
    from matplotlib.colors import colorConverter
    color, bg = map(lambda c: np.array(colorConverter.to_rgb(c)), (color, bg))
    return (1 - blend) * bg + blend * color


def make_cmap_and_norm(data, colors, blend=1):
    if not isinstance(colors, (list, tuple)):
        colors = [colors]
    if blend < 1:
        colors = [blend_colors(c, 'white', blend) for c in colors]

    # colormap with an boundary norm to match the unique data points
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)
    boundaries = np.append(np.unique(data), np.inf)
    norm = BoundaryNorm(boundaries, len(boundaries) - 1)

    return cmap, norm


def plot_hoppings(ax, positions, hoppings, width,
                  offset=(0, 0, 0), boundary=False, blend=1, **kwargs):
    if width == 0:
        return

    kwargs = with_defaults(kwargs, zorder=-1, colors='#666666')

    colors = kwargs.pop('colors')
    if colors == 'default':
        colors = ["#666666", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    kwargs['cmap'], kwargs['norm'] = make_cmap_and_norm(hoppings.values, colors, blend)

    ndims = 3 if ax.name == '3d' else 2
    offset = np.array(offset[:ndims])
    positions = np.array(positions[:ndims]).T

    if not boundary:
        pos = positions + offset
        lines = ((pos[i], pos[j]) for i, j in hoppings.indices())
    else:
        from itertools import chain
        lines = chain(
            ((positions[i] + offset, positions[j]) for i, j in hoppings.indices()),
            ((positions[i], positions[j] - offset) for i, j in hoppings.indices())
        )

    if ndims == 2:
        from matplotlib.collections import LineCollection
        col = LineCollection(lines, lw=width, **kwargs)
        col.set_array(hoppings.values.copy())
        ax.add_collection(col)
        ax.autoscale_view()
    else:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        had_data = ax.has_data()
        col = Line3DCollection(list(lines), lw=width, **kwargs)
        col.set_array(hoppings.values.copy())
        ax.add_collection3d(col)

        ax.set_zmargin(0.5)
        minmax = np.vstack((positions.min(axis=0), positions.max(axis=0))).T
        ax.auto_scale_xyz(*minmax, had_data=had_data)

    return col


def plot_sites(ax, positions, data, radius, offset=(0, 0, 0), blend=1, **kwargs):
    if np.all(radius == 0):
        return

    kwargs = with_defaults(kwargs, alpha=0.95, lw=0.1)

    # create colormap from discrete colors
    if 'cmap' not in kwargs:
        colors = kwargs.pop('colors', None)
        if not colors or colors == 'default':
            colors = ["#377ec8", "#ff7f00", "#41ae76", "#e41a1c",
                      "#984ea3", "#ffff00", "#a65628", "#f781bf"]
        elif colors == 'pairs':
            colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
                      "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
        kwargs['cmap'], kwargs['norm'] = make_cmap_and_norm(data, colors, blend)

    # create array of (x, y) points
    points = np.array(positions[:2]).T + offset[:2]

    if ax.name != '3d':
        # sort based on z position to get proper 2D z-order
        idx = positions[2].argsort()
        if not np.isscalar(radius):
            radius = radius[idx]
        points, data = points[idx], data[idx]

        from pybinding.support.collections import CircleCollection
        col = CircleCollection(radius, offsets=points, transOffset=ax.transData, **kwargs)
        col.set_array(data)

        ax.add_collection(col)
        ax.autoscale_view()
    else:
        from pybinding.support.collections import Circle3DCollection
        col = Circle3DCollection(radius/8, offsets=points, transOffset=ax.transData, **kwargs)
        col.set_array(data)
        z = positions[2] + offset[2]
        col.set_3d_properties(z, 'z')

        had_data = ax.has_data()
        ax.add_collection(col)
        minmax = tuple((v.min(), v.max()) for v in positions)
        ax.auto_scale_xyz(*minmax, had_data=had_data)

    return col


def plot_site_indices(system):
    # show the Hamiltonian index next to each atom (for debugging)
    from pybinding.plot.annotate import annotate_box
    for i, xy in enumerate(zip(system.x, system.y)):
        annotate_box(i, xy)


def plot_hopping_values(system):
    from pybinding.plot.annotate import annotate_box
    pos = np.array(system.positions[:2]).T

    for i, j, t in system.matrix.triplets():
        annotate_box(t, (pos[i] + pos[j]) / 2)

    for boundary in system.boundaries:
        from pybinding.support.sparse import SparseMatrix
        hoppings = boundary.matrix
        hoppings.__class__ = SparseMatrix

        for i, j, t in hoppings.triplets():
            annotate_box(t, (pos[i] + pos[j] + boundary.shift[:2]) / 2)
            annotate_box(t, (pos[i] + pos[j] - boundary.shift[:2]) / 2)
