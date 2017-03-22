Material Repository
===================

The repository includes a few common lattices, shapes, fields and other kinds of helpful functions
and constants. A material can be imported from `pybinding.repository`, for example::

    from pybinding.repository import graphene

    lattice = graphene.monolayer()

Or::

    from pybinding.repository import phosphorene

    lattice = phosphorene.monolayer_4band()


.. toctree::
    :caption: Materials
    :maxdepth: 1

    graphene
    phosphorene
    group6_tmd
