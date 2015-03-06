import matplotlib.pyplot as _plt


def annotate_box(s, xy, fontcolor='black', alpha=0.5, lw=0.3, **kwargs):
    defaults = dict(horizontalalignment='center', verticalalignment='center')
    bbox = dict(boxstyle="round,pad=0.2", alpha=alpha, lw=lw,
                fc='white' if fontcolor != 'white' else 'black')
    properties = dict(defaults, color=fontcolor, bbox=bbox, **kwargs)
    _plt.annotate(s, xy, **properties)
