import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .utils import with_defaults


def _get_ax(ax=None):
    return ax if ax else plt.gca()


def despine(trim=False, ax=None):
    """Remove the top and right spines

    Parameters
    ----------
    trim : bool
        Trim spines so that they don't extend beyond the last major ticks.
    """
    ax = _get_ax(ax)
    if ax.name == '3d':
        return

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    if trim:
        for v, side in [('x', 'bottom'), ('y', 'left')]:
            ax.spines[side].set_smart_bounds(True)
            ticks = getattr(ax, "get_{}ticks".format(v))()
            vmin, vmax = getattr(ax, "get_{}lim".format(v))()
            ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
            getattr(ax, "set_{}ticks".format(v))(ticks)


def despine_all(ax=None):
    """Remove all spines, axes labels and ticks
    """
    ax = _get_ax(ax)
    if ax.name == '3d':
        return

    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])


def set_min_range(min_range, vs='xy', ax=None):
    """Set minimum axis range"""
    ax = _get_ax(ax)
    for v in vs:
        vmin, vmax = getattr(ax, "get_{}lim".format(v))()
        if abs(vmax - vmin) < min_range:
            c = (vmax + vmin) / 2
            vmin, vmax = c - min_range / 2, c + min_range / 2
            getattr(ax, "set_{}lim".format(v))(vmin, vmax, auto=None)


def add_margin(margin=0.08, vs='xy', ax=None):
    """Adjust the axis range to include a margin (after autoscale)"""
    ax = _get_ax(ax)
    for v in vs:
        vmin, vmax = getattr(ax, "get_{}lim".format(v))()
        set_min_range(abs(vmax - vmin) * (1 + margin), vs=v, ax=ax)


def blend_colors(color, bg, factor):
    """Blend color with background"""
    from matplotlib.colors import colorConverter
    color, bg = map(lambda c: np.array(colorConverter.to_rgb(c)), (color, bg))
    return (1 - factor) * bg + factor * color


def colorbar(mappable=None, cax=None, ax=None, powerlimits=(0, 0), **kwargs):
    """Convenient colorbar function"""
    cbar = plt.colorbar(mappable, cax, ax, **with_defaults(kwargs, pad=0.02, aspect=28))
    cbar.solids.set_edgecolor("face")  # remove white gaps between segments
    if powerlimits and hasattr(cbar.formatter, 'set_powerlimits'):
        cbar.formatter.set_powerlimits(powerlimits)
    cbar.update_ticks()

    return cbar


def annotate_box(s, xy, fontcolor='black', **kwargs):
    """Annotate with a box around the text"""
    kwargs['bbox'] = with_defaults(
        kwargs.get('bbox', {}),
        boxstyle="round,pad=0.2", alpha=0.5, lw=0.3,
        fc='white' if fontcolor != 'white' else 'black'
    )

    if all(key in kwargs for key in ['arrowprops', 'xytext']):
        kwargs['arrowprops'] = with_defaults(
            kwargs['arrowprops'], dict(arrowstyle="->", color=fontcolor)
        )

    plt.annotate(s, xy, **with_defaults(kwargs, color=fontcolor, horizontalalignment='center',
                                        verticalalignment='center'))


def cm2inch(*values):
    """ Convert from centimeter to inch """
    return tuple(v / 2.54 for v in values)


def legend(*args, reverse=False, facecolor='0.98', lw=0, **kwargs):
    if not reverse:
        ret = plt.legend(*args, **kwargs)
    else:
        h, l = plt.gca().get_legend_handles_labels()
        ret = plt.legend(h[::-1], l[::-1], *args, **kwargs)

    frame = ret.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_linewidth(lw)
    return ret


def get_palette(name=None, num_colors=8, start=0):
    if not name:
        return mpl.rcParams["axes.color_cycle"]

    brewer = dict(Set1=9, Set2=8, Set3=12, Pastel1=9, Pastel2=8, Accent=8, Dark2=8, Paired=12)
    if name in brewer:
        total = brewer[name]
        take = min(num_colors, total)
        bins = np.linspace(0, 1, total)[:take]
    else:
        bins = np.linspace(0, 1, num_colors + 2)[1:-1]

    cmap = plt.get_cmap(name)
    palette = cmap(bins)[:, :3]

    from itertools import cycle, islice
    palette = list(islice(cycle(palette), start, start + num_colors))
    return palette


def set_palette(name=None, num_colors=8, start=0):
    palette = get_palette(name, num_colors, start)
    mpl.rcParams["axes.color_cycle"] = list(palette)
    mpl.rcParams["patch.facecolor"] = palette[0]


def direct_cmap_norm(data, colors, blend=1):
    """Colormap with direct mapping: data[i] -> colors[i]"""
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
