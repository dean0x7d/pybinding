Phosphorene
===========

.. meta::
   :description: Collection of phosphorene lattices, constants and fields for pybinding

.. module:: pybinding.repository.phosphorene

.. plot::
    :context: reset
    :nofigs:
    :include-source: False

    from pybinding.repository.phosphorene import *

.. autofunction:: monolayer_4band

    .. plot::
        :context: close-figs
        :include-source: False
        :alt: Phosphorene unit cell

        plt.figure(figsize=(7, 4))
        monolayer_4band(2).plot()
        plt.title("monolayer_4band(num_hoppings=2)")

    .. plot::
        :context: close-figs
        :include-source: False
        :alt: Phosphorene unit cell

        plt.figure(figsize=(7, 4))
        monolayer_4band(5).plot()
        plt.title("monolayer_4band(num_hoppings=5)")
