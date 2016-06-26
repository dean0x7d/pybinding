Green's function
================

.. meta::
   :description: Computing Green's function of a tight-binding Hamiltonian matrix

Green's function methods were used briefly in the :doc:`fields` and :doc:`strain` sections. As with
the eigensolvers, there is one common :class:`.Greens` interface while the underlying algorithm
may be implemented in various ways. At this time, :func:`.kpm` is the only one that comes with
the package.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


KPM
---

The :func:`.kpm` implementation is a very efficient way of calculating Green’s function, especially
for large sparse Hamiltonian matrices. Based on the kernel polynomial method, the approach
approximates the Green's function by expanding it into a series of Chebyshev polynomials.
For more information, see the references `linked here <http://arxiv.org/abs/cond-mat/0504627>`_
and `here <http://arxiv.org/abs/1007.1609>`_.

A great advantage of this method is that memory usage and computation time scale linearly with
problem dimension. In addition, computation time can be tuned based on the required accuracy
which is conveniently expressed as a Lorentzian broadening width. Finally, each spatial site can
be computed separately which means local properties can be calculated efficiently at a fraction
of the time required for the entire system.


Greens interface
----------------

The interface is quite simple. A :class:`.Greens` function is created with the desired
implementation:

.. code-block:: python
    :emphasize-lines: 0

    model = pb.Model(graphene.monolayer())
    greens = pb.greens.kpm(model)

It can then be used to calculate the Green's function corresponding to Hamiltonian matrix element
`i,j` for the desired energy range and broadening:

.. code-block:: python
    :emphasize-lines: 0

    g_ij = greens(i, j, energy=np.linspace(-9, 9, 100), broadening=0.1)

The result is the raw Green's function data for the given matrix element. However, there is also
a convenient :meth:`.Greens.calc_ldos` method which makes it very easy to calculate the local
density of states (LDOS). In the next example we'll use a large square sheet of pristine graphene:

.. plot::
    :context: close-figs
    :alt: Graphene density of states

    from pybinding.repository import graphene

    model = pb.Model(
        graphene.monolayer(),
        pb.rectangle(60)
    )
    greens = pb.greens.kpm(model)
    ldos = greens.calc_ldos(energy=np.linspace(-9, 9, 200), broadening=0.05, position=[0, 0])
    ldos.plot()

The LDOS is calculated for energies between -9 and 9 eV with a Lorentzian broadening of 50 meV.
Since this is the *local* density of states, position is also a required argument. We target the
center of our square system where we expect to see the well-known LDOS shape of pristine graphene.
Indeed, that is what the resulting :class:`.LDOS` object shows after invoking its
:meth:`~.LDOS.plot` method.

Tight-binding systems have lattice sites at discrete positions, which in principle means that we
cannot freely choose just any position for LDOS calculations. However, as a convenience the
:meth:`.Greens.calc_ldos` method will automatically find a valid site closest to the given target
position. We can optionally also choose a specific sublattice:

.. code-block:: python
    :emphasize-lines: 0

    ldos = greens.calc_ldos(energy=np.linspace(-9, 9, 200), broadening=0.05,
                            position=[0, 0], sublattice='B')

In this case we would calculate the LDOS at a site of sublattice B closest to the center of the
system. We can try that on a graphene system with a mass term:

.. plot::
    :context: close-figs
    :alt: Graphene density of states (with mass term induced by a substrate)

    model = pb.Model(
        graphene.monolayer(),
        graphene.mass_term(1),
        pb.rectangle(60)
    )
    greens = pb.greens.kpm(model)

    for sub_name in ['A', 'B']:
        ldos = greens.calc_ldos(energy=np.linspace(-9, 9, 500), broadening=0.05,
                                position=[0, 0], sublattice=sub_name)
        ldos.plot(label=sub_name)
    pb.pltutils.legend()

Multiple plots compose nicely here. A large band gap is visible at zero energy due to the inclusion
of :func:`graphene.mass_term() <.graphene.modifiers.mass_term>`. It places an onsite potential with
the opposite sign in each sublattice. This is also why the LDOS lines for A and B sublattices are
antisymmetric around zero energy with respect to one another.


Further reading
---------------

For an additional examples see the :ref:`magnetic-field-modifier` subsection of :doc:`fields` as
well as the :ref:`Strain modifier <strain-modifier>` subsection of :doc:`strain`.
The reference page for the :mod:`.greens` submodule contains more information.
