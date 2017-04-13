Imports
-------

This tutorial includes two kinds of example code: complete files and short code samples. Files
are self-contained examples which can be downloaded and run. Code snippets are included directly
within the tutorial text to illustrate features, thus they omit some common and repetitive code
(like import statements) in order to save space and not distract from the main point. It is
assumed that the following lines precede any other code::

    import pybinding as pb
    import numpy as np
    import matplotlib.pyplot as plt

    pb.pltutils.use_style()

The `pb` alias is always used for importing pybinding. This is similar to the common scientific
package aliases: `np` and `plt`. These import conventions are used consistently in the tutorial.

The function :func:`pb.pltutils.use_style() <.pltutils.use_style()>` applies pybinding's default
style settings for matplotlib. This is completely optional and only affects the aesthetics of the
generated figures.
