Phosphorene
===========

.. plot::
    :context: reset
    :nofigs:
    :include-source: False

    from pybinding.repository.phosphorene import *

.. automodule:: pybinding.repository.phosphorene
    :members:
    :exclude-members: monolayer_4band

    .. autofunction:: monolayer_4band

        .. plot::
            :context: close-figs
            :include-source: False

            plt.figure(figsize=(8, 4))
            monolayer_4band().plot()