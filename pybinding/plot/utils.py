import numpy as np
import matplotlib.pyplot as plt
from pybinding.utils import with_defaults


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
    cbar = plt.colorbar(mappable, cax, ax, **kwargs)
    cbar.solids.set_edgecolor("face")  # remove white gaps between segments
    if powerlimits:
        cbar.formatter.set_powerlimits(powerlimits)
    cbar.update_ticks()


def annotate_box(s, xy, fontcolor='black', alpha=0.5, lw=0.3, **kwargs):
    """Annotate with a box around the text"""
    bbox = dict(boxstyle="round,pad=0.2", alpha=alpha, lw=lw,
                fc='white' if fontcolor != 'white' else 'black')
    plt.annotate(s, xy, **with_defaults(kwargs, color=fontcolor, bbox=bbox,
                                        horizontalalignment='center', verticalalignment='center'))
