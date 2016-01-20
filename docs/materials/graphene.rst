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

            monolayer().plot()


    .. autofunction:: monolayer_alt

        .. plot::
            :context: close-figs
            :include-source: False

            monolayer_alt().plot()


    .. autofunction:: monolayer_4atom

        .. plot::
            :context: close-figs
            :include-source: False

            plt.figure(figsize=(5, 5))
            monolayer_4atom().plot()
            plt.xlim(-0.5, 0.5)
            plt.xticks([-0.4, -0.2, 0, 0.2, 0.4])

    .. autofunction:: monolayer_nn

        .. plot::
            :context: close-figs
            :include-source: False

            monolayer_nn().plot()

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


Fields
------

.. automodule:: pybinding.repository.graphene.electric
    :members:

.. automodule:: pybinding.repository.graphene.magnetic
    :members:

Strain
------

.. automodule:: pybinding.repository.graphene.strain
    :members:
