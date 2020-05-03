API Reference
=============

This detailed reference lists all the classes and functions contained in the package.
If you are just looking to get started, read the :doc:`/tutorial/index` first.

The :class:`.Lattice` describes the unit cell of a crystal, while the :class:`.Model` is used to
build up a larger system by translating the unit cell to fill a certain shape or symmetry.
The model builds the Hamiltonian matrix by applying fields and other modifier parameters.

.. currentmodule:: pybinding

.. autosummary::
    :toctree: _api

    Lattice
    Model

.. _shapes_api:

Shapes
------

The geometry of a finite-sized system can be defined using the :class:`.Polygon` class (2D only)
or using :class:`.FreeformShape` (1 to 3 dimensions). A few common shapes are included in the
package and listed below. These predefined shapes are just functions which configure and return
a shape class object.

.. rubric:: Building blocks

.. autosummary::
    :toctree: _api

    Polygon
    FreeformShape
    CompositeShape

.. rubric:: Predefined shapes

.. autosummary::
    :toctree: _api

    circle
    line
    primitive
    rectangle
    regular_polygon


Symmetry
--------

.. autosummary::
    :toctree: _api

    translational_symmetry


.. _modifiers_api:

Modifiers
---------

The following decorators are used to create functions which express some feature of a
tight-binding model, such as various fields, defects or geometric deformations.

.. rubric:: Decorators

.. autosummary::
    :toctree: _api

    site_state_modifier
    site_position_modifier
    onsite_energy_modifier
    hopping_energy_modifier


.. rubric:: Predefined modifiers

.. autosummary::
    :toctree: _api

    constant_potential
    force_double_precision
    force_complex_numbers


.. _generators_api:

Generators
----------

The following decorators are used to create functions which add features to a
tight-binding model, such as hoppings and sites.

.. rubric:: Decorators

.. autosummary::
    :toctree: _api

    hopping_generator
    site_generator


.. _compute_api:

Compute
-------

After a :class:`.Model` is constructed, computational routines can be applied to determine various
physical properties. The following submodules contain functions for exact diagonalization as well
as some approximative compute methods. Follow the links below for details.

.. autosummary::
    :toctree: _api

    solver
    chebyshev

.. rubric:: Experimental

.. autosummary::
    :toctree: _api

    parallel

.. _results_api:

Results
-------

Result objects are usually produced by compute functions, but they are also used to express certain
model properties. They hold data and offer postprocessing and plotting methods specifically adapted
to the nature of the physical properties (i.e. the stored data).

The utility functions :func:`pb.save() <.save>` and :func:`pb.load() <.load>` can be used to
efficiently store entire result objects into files. The information about the kind of physical
property is saved along with the raw data, i.e. executing `result = pb.load("data_file.pbz")`
followed by `result.plot()` will work and present the appropriate figure.

.. autosummary::
    :toctree: _api

    save
    load
    make_path
    Bands
    Eigenvalues
    Series
    SpatialMap
    StructureMap
    Sweep
    NDSweep


Components
----------

The following submodules contain classes and functions which are not meant to created manually,
but they are components of other classes (e.g. :class:`Model`) so they are used regularly (even
if indirectly).

.. autosummary::
    :toctree: _api

    system
    leads


Miscellaneous
-------------

.. autosummary::
    :toctree: _api

    constants
    pltutils
