Graphene
========

.. automodule:: pybinding.repository.graphene
    :members:

Lattices
--------

.. plot::
    :context: reset
    :nofigs:
    :include-source: False

    from pybinding.repository.graphene import *

.. automodule:: pybinding.repository.graphene.lattice

    .. autofunction:: monolayer

        .. plot::
            :context: close-figs
            :include-source: False

            monolayer(nearest_neighbors=1).plot()
            plt.title('monolayer(nearest_neighbors=1)')

        .. plot::
            :context: close-figs
            :include-source: False

            monolayer(nearest_neighbors=2).plot()
            plt.title('monolayer(nearest_neighbors=2)')

        .. plot::
            :context: close-figs
            :include-source: False

            plt.figure(figsize=(5, 5.5))
            monolayer(nearest_neighbors=3).plot()
            plt.title('monolayer(nearest_neighbors=3)')

    .. autofunction:: monolayer_4atom

        .. plot::
            :context: close-figs
            :include-source: False

            plt.figure(figsize=(5, 5))
            monolayer_4atom().plot()
            plt.xlim(-0.5, 0.5)
            plt.xticks([-0.4, -0.2, 0, 0.2, 0.4])

    .. autofunction:: bilayer

        .. plot::
            :context: close-figs
            :include-source: False

            plt.figure(figsize=(3.8, 3.8))
            bilayer().plot()

Constants
---------

.. automodule:: pybinding.repository.graphene.constants
    :members:

Shapes
------

.. automodule:: pybinding.repository.graphene.shape
    :members:


Modifiers
---------

.. automodule:: pybinding.repository.graphene.modifiers
    :members:
