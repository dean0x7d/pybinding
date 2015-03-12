import matplotlib.pyplot as plt


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
