Band structure
--------------

In order to calculate the band structure of a crystal lattice, this section is going to introduce
the concepts of a :class:`.Model` and a :class:`.Solver`.


Model
*****

A :class:`.Model` contains the full tight-binding description of the physical system that we wish
to solve. We'll start by assigning a lattice to the model, and we'll use a pre-made one from the
material repository.

.. plot::
    :context: reset

    from pybinding.repository import graphene

    model = pb.Model(graphene.monolayer())
    model.system.plot()

The result is not very exciting: just a single graphene unit cell, with 2 atoms and a single
hopping between them. The model does not assume translational symmetry or any other physical
property. Given a lattice it will just create a single unit cell. The model has a :class:`.System`
attribute which keeps track of structural properties like the positions of lattice sites and the
way they are connected, as seen in the figure above.

The model also has a :attr:`.hamiltonian` attribute::

    >>> model.hamiltonian
    (0, 1)   -2.8
    (1, 0)   -2.8

It's a matrix (in the `scipy.sparse.csr_matrix` format) which corresponds to the tight-binding
Hamiltonian of our model. The output above shows the default sparse representation of the data
with `(row, col) value`. Alternatively, we can see the dense matrix output::

    >>> model.hamiltonian.todense()
    [[ 0.0 -2.8]
     [-2.8  0.0]]

Next, we include :func:`.translational_symmetry` to create an infinite graphene sheet.

.. plot::
    :context: close-figs

    model = pb.Model(
        graphene.monolayer(),
        pb.translational_symmetry()
    )
    model.system.plot()

The red lines indicate hoppings on periodic boundaries. The lighter colored circles represent the
translations of the unit cell. The number of translations is infinite, but the plot only presents
the first one in each lattice vector direction.


Solver
******

A :class:`.Solver` can exactly calculate the eigenvalues and eigenvectors of a Hamiltonian matrix.
We'll start by using a :func:`.lapack` solver which is the simplest and most appropriate for small
systems.

    >>> model = pb.Model(graphene.monolayer())
    >>> solver = pb.solver.lapack(model)
    >>> solver.eigenvalues
    [-2.8 2.8]
    >>> solver.eigenvectors
    [[-0.707 -0.707]
     [-0.707  0.707]]

Beyond just the `eigenvalues` and `eigenvectors` attributes, :class:`.Solver` has a convenient
:meth:`.calc_bands` method.

.. plot::
    :context: close-figs

    from math import sqrt, pi

    model = pb.Model(
        graphene.monolayer(),
        pb.translational_symmetry()
    )
    solver = pb.solver.lapack(model)

    a_cc = graphene.a_cc
    Gamma = [0, 0]
    K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
    M = [0, 2*pi / (3*a_cc)]
    K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

    bands = solver.calc_bands(K1, Gamma, M, K2)
    bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])

The points :math:`\Gamma, K` and :math:`M` are used to draw a path in the reciprocal space of
graphene's Brillouin zone and :meth:`.calc_bands` calculates the band structure along that path.
The return value of the method is a :class:`.Bands` result object.


Example
*******

:download:`Download source code </tutorial/bands_example.py>`

.. plot:: tutorial/bands_example.py
    :include-source:


Further reading
***************

For more band structure calculations check out the :doc:`examples section </examples/lattice/index>`.
