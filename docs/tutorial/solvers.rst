Eigenvalue solvers
==================

.. meta::
   :description: Computing the eigenvalues and eigenvectors of a tight-binding Hamiltonian matrix

Solvers were first introduced in the :doc:`bands` section and then used throughout the tutorial to
present the results of the various models we constructed. This section will take a more detailed
look at the concrete :func:`.lapack` and :func:`.arpack` eigenvalue solvers and their common
:class:`.Solver` interface.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


LAPACK
------

The :class:`.Solver` class establishes the interface of a solver within pybinding, but it does not
contain a concrete diagonalization routine. For this reason we never instantiate the plain solver,
only its implementations such as :func:`.solver.lapack`.

The LAPACK implementation works on dense matrices which makes it well suited only for small
systems. However, a great advantage of this solver is that it always solves for all eigenvalues
and eigenvectors of a Hamiltonian matrix. This makes it perfect for calculating the entire band
structure of the bulk or nanoribbons, as has been shown several times in this tutorial.

Internally, this solver uses the :func:`scipy.linalg.eigh` function for dense Hermitian matrices.
See the :func:`.solver.lapack` API reference for more details.


ARPACK
------

The :func:`.solver.arpack` implementation works on sparse matrices which makes it suitable for
large systems. However, only a small subset of the total eigenvalues and eigenvectors can be
calculated. This tutorial already contains a few examples where the ARPACK solver was used, and
one more is presented below.

Internally, the :func:`scipy.sparse.linalg.eigsh` function is used to solve large sparse Hermitian
matrices. The first argument to :func:`.solver.arpack` must be the pybinding :class:`.Model`, but
the following arguments are the same as :func:`~scipy.sparse.linalg.eigsh`, so the solver routine
can be tweaked as desired. Rather than reproduce the full list of options here, we refer you to
the scipy :func:`~scipy.sparse.linalg.eigsh` reference documentation. Here, we will focus on the
specific features of solvers within pybinding.


Solver interface
----------------

No matter which concrete solver is used, they all share a common :class:`.Solver` interface.
The two primary properties are :attr:`~.Solver.eigenvalues` and :attr:`~.Solver.eigenvectors`.
These are the raw results of the exact diagonalization of the Hamiltonian matrix.

    >>> from pybinding.repository import graphene
    >>> model = pb.Model(graphene.monolayer())
    >>> model.hamiltonian.todense()
    [[ 0.0 -2.8]
     [-2.8  0.0]]
    >>> solver = pb.solver.lapack(model)
    >>> solver.eigenvalues
    [-2.8 2.8]
    >>> solver.eigenvectors
    [[-0.707 -0.707]
     [-0.707  0.707]]

The properties contain just the raw data. However, :class:`.Solver` also offers a few convenient
calculation methods. We'll demonstrate these on a simple rectangular graphene system.

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    model = pb.Model(
        graphene.monolayer(),
        pb.rectangle(x=3, y=1.2)
    )
    model.plot()

First, we'll take a look at the :meth:`~.Solver.calc_eigenvalues` method. While its job is
essentially the same as the :attr:`~.Solver.eigenvalues` property, there is one key difference:
the property returns a raw array, while the method returns an :class:`.Eigenvalues` result object.
These objects have convenient functions built in and they know how to plot their data:

.. plot::
    :context: close-figs
    :alt: Energy states of a graphene quantum dot

    solver = pb.solver.arpack(model, k=20)  # for the 20 lowest energy eigenvalues
    eigenvalues = solver.calc_eigenvalues()
    eigenvalues.plot()

The basic plot just shows the state number and energy of each eigenstate, but we can also do
something more interesting. If we pass a position argument to :meth:`~.Solver.calc_eigenvalues`
it will calculate the probability density :math:`|\Psi(\vec{r})|^2` at that position for each
eigenstate and we can view the result using :meth:`.Eigenvalues.plot_heatmap`:

.. plot::
    :context: close-figs
    :alt: Energy states of a graphene quantum dot with probability heatmap

    eigenvalues = solver.calc_eigenvalues(map_probability_at=[0.1, 0.6])  # position in [nm]
    eigenvalues.plot_heatmap(show_indices=True)
    pb.pltutils.colorbar()

In this case we are interested in the probability density at `[x, y] = [0.1, 0.6]`, i.e. a lattice
site at the top zigzag edge of our system. Note that the given position does not need to be
precise: the probability will be computed for the site closest to the given coordinates. From the
figure we can see that the probability at the edge is highest for the two zero-energy states:
numbers 9 and 10. We can take a look at the spatial map of state 9 using the
:meth:`~.Solver.calc_probability` method:

.. plot::
    :context: close-figs
    :alt: Spatial map of the probability density of a graphene quantum dot

    probability_map = solver.calc_probability(9)
    probability_map.plot()

The result object in this case is a :class:`.StructureMap` with the probability density
:math:`|\Psi(\vec{r})|^2` as its data attribute. As expected, the most prominent states are at
the zigzag edges of the system.

An alternative way to get a spatial map of the system is via the local density of states (LDOS).
The :meth:`~.Solver.calc_spatial_ldos` method makes this easy. The LDOS map is requested for a
specific energy value instead of a state number and it considers multiple states within a Gaussian
function with the specified broadening:

.. plot::
    :context: close-figs
    :alt: Spatial LDOS of a graphene quantum dot

    ldos_map = solver.calc_spatial_ldos(energy=0, broadening=0.05)  # [eV]
    ldos_map.plot()

The total density of states can be calculated with :meth:`~.Solver.calc_dos`:

.. plot::
    :context: close-figs
    :alt: Total density of states (DOS) of a graphene quantum dot

    dos = solver.calc_dos(energies=np.linspace(-1, 1, 200), broadening=0.05)  # [eV]
    dos.plot()

Our example system is quite small so the DOS does not resemble bulk graphene. The zero-energy peak
stands out as the signature of the zigzag edge states.

.. _manual_band_calculation:

For periodic systems, the wave vector can be controlled using :meth:`.Solver.set_wave_vector`.
This allows us to compute the eigenvalues at various points in k-space. For example:

.. plot::
    :context: close-figs
    :alt: Graphene band structure

    from math import pi

    model = pb.Model(
        graphene.monolayer(),
        pb.translational_symmetry()
    )
    solver = pb.solver.lapack(model)

    kx_lim = pi / graphene.a
    kx_path = np.linspace(-kx_lim, kx_lim, 100)
    ky_outer = 0
    ky_inner = 2*pi / (3*graphene.a_cc)

    outer_bands = []
    for kx in kx_path:
        solver.set_wave_vector([kx, ky_outer])
        outer_bands.append(solver.eigenvalues)

    inner_bands = []
    for kx in kx_path:
        solver.set_wave_vector([kx, ky_inner])
        inner_bands.append(solver.eigenvalues)

    for bands in [outer_bands, inner_bands]:
        result = pb.results.Bands(kx_path, bands)
        result.plot()

This example shows the basic principle of iterating over a path in k-space in order to calculate
the band structure. However, this is made much easier with the :meth:`.Solver.calc_bands` method.
This was already covered in the :doc:`bands` section and will not be repeated here. But keep in
mind that this calculation does not need to be done manually, :meth:`.Solver.calc_bands` is the
preferred way.


Further reading
---------------

Take a look at the :mod:`.solver` and :mod:`.results` reference pages for more detailed
information. More solver examples are available throughout this tutorial.
