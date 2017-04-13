Plotting Guide
==============

All of the plotting functions in pybinding are create using `matplotlib <http://matplotlib.org/>`_.
This means that you can customize the appearance of the figures using standard matplotlib commands.
However, some plots (like lattice structure) are specialized to tight-binding models and have some
additional options in contrast to ordinary plot templates (line plot, scatter, quiver, etc.). This
guide will present the workflow for customizing figures in pybinding.

You can also create your own figures from scratch using just the raw data from pybinding. However,
it is far more convenient to use pybinding's builtin plot methods as a base and use matplotlib's
API to customize as needed. The builtin methods have already taken care of most of the work needed
to represent arbitrary tight-binding models and their properties. This is done in the most general
way possible in order to produce reasonable looking figures for most systems. However, because of
the huge variety of tight-binding models, the preset style may not always be ideal. This is where
this customization guide comes in.


.. toctree::
    :maxdepth: 1

    structure
    structuremap
