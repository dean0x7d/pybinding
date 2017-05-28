Kwant compatibility
===================

`Kwant <http://kwant-project.org/>`_ is a Python package for numerical tight-binding similar to
pybinding, but it's specialized for transport calculations. Since the two packages work with the
same kind of Hamiltonian matrices, it's possible to build a model in pybinding and use Kwant to
compute the transport properties. The advantage for pybinding users is access to Kwant's transport
solvers in addition to pybinding's builtin :ref:`computational routines <compute_api>`. The
advantage for Kwant users is the much faster system build times made possible by pybinding's model
builder -- see the :doc:`/benchmarks/index`.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


Exporting a model
-----------------

The procedure for constructing and solving transport problems in Kwant can be summarized with
the following lines of pseudo-code:

.. code-block:: python
    :emphasize-lines: 0

    # 1. BUILD model system
    builder = kwant.Builder()
    ...  # specify model parameters
    system = builder.finalized()

    # 2. COMPUTE scattering matrix
    smatrix = kwant.smatrix(system)
    ...  # call smatrix methods

If we want to use pybinding to build the model, we can just replace the first part:

.. code-block:: python
    :emphasize-lines: 0

    # 1. BUILD model system
    model = pb.Model(...)  # specify model parameters
    kwant_system = model.tokwant()  # export to kwant format

    # 2. COMPUTE scattering matrix
    smatrix = kwant.smatrix(kwant_system)
    ...  # call smatrix methods

A pybinding :class:`.Model` is defined as usual and then converted to the Kwant-compatible format
by calling the :meth:`.Model.tokwant` method. The resulting `kwant_system` can be used as expected.


Complete example
----------------

A detailed overview of scattering model construction in pybinding is available in the
:doc:`tutorial </tutorial/scattering>`. Here, we present a simple example of a graphene wire
with a potential barrier:

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    def potential_barrier(v0, x0):
        """Barrier height `v0` in eV with spatial position `-x0 <= x <= x0`"""
        @pb.onsite_energy_modifier(is_double=True)  # enable double-precision floating-point
        def function(energy, x):
            energy[np.logical_and(-x0 <= x, x <= x0)] = v0
            return energy
        return function

    def make_model(length, width, v0=0):
        model = pb.Model(
            graphene.monolayer(),
            pb.rectangle(length, width),
            potential_barrier(v0, length / 4)
        )
        model.attach_lead(-1, pb.line([-length/2, -width/2], [-length/2, width/2]))
        model.attach_lead(+1, pb.line([ length/2, -width/2], [ length/2, width/2]))
        return model

    model = make_model(length=1, width=2)  # nm
    model.plot()

We can then vary the height of the potential barrier and calculate the transmission using Kwant:

.. code-block:: python

    import kwant

    length, width = 15, 15  # nm
    electron_energy = 0.25  # eV
    barrier_heights = np.linspace(0, 0.5, 100)  # eV

    transmission = []
    for v in barrier_heights:
        model = make_model(length, width, v)  # pybinding model
        kwant_system = model.tokwant()  # export to kwant
        smatrix = kwant.smatrix(kwant_system, energy=electron_energy)
        transmission.append(smatrix.transmission(1, 0))

For more information about `kwant.smatrix` and other transport calculations, please refer to the
`Kwant website <http://kwant-project.org/>`_. That is outside the scope of this guide. The purpose
of this section is to present the :meth:`.Model.tokwant` compatibility method. The exported system
is then in the domain of Kwant.

From there, it's trivial to plot the results:

.. plot::
    :nofigs:
    :context: close-figs
    :include-source: False

    electron_energy = 0.25
    barrier_heights = np.linspace(0, 0.5, 100)
    transmission = [3.00, 3.00, 2.99, 2.98, 2.96, 2.94, 2.90, 2.84, 2.78, 2.70, 2.60, 2.50, 2.39,
                    2.27, 2.16, 2.04, 1.94, 1.83, 1.74, 1.66, 1.58, 1.52, 1.46, 1.41, 1.36, 1.32,
                    1.29, 1.26, 1.23, 1.21, 1.19, 1.17, 1.16, 1.14, 1.13, 1.12, 1.11, 1.10, 1.09,
                    1.08, 1.08, 1.07, 1.07, 1.06, 1.06, 1.05, 1.05, 1.04, 1.04, 1.03, 0.61, 0.67,
                    0.39, 1.12, 0.78, 0.67, 0.58, 0.52, 0.54, 0.68, 0.89, 1.02, 0.99, 0.89, 0.80,
                    0.74, 0.71, 0.70, 0.69, 0.70, 0.71, 0.74, 0.77, 0.82, 0.87, 0.93, 0.98, 1.02,
                    1.03, 1.01, 0.97, 0.92, 0.86, 0.82, 0.79, 0.78, 0.79, 0.81, 0.85, 0.90, 0.96,
                    1.03, 1.11, 1.18, 1.27, 1.35, 1.44, 1.52, 1.60, 1.67]

    plt.figure(figsize=(3, 2.4))

.. plot::
    :context:

    plt.plot(barrier_heights, transmission)
    plt.ylabel("transmission")
    plt.xlabel("barrier height (eV)")
    plt.axvline(electron_energy, 0, 0.5, color="gray", linestyle=":")
    plt.annotate("electron energy\n{} eV".format(electron_energy), (electron_energy, 0.54),
                 xycoords=("data", "axes fraction"), horizontalalignment="center")
    pb.pltutils.despine()  # remove top and right axis lines

Note that the transmission was calculated for an energy value of 0.25 eV. As the height of the
barrier is increased, two regimes are clearly distinguishable: transmission over and through the
barrier.


Performance considerations
--------------------------

The Kwant documentation recommends separating model parameters into two parts: the structural data
which remains constant and fields which can be varied. This yields better performance because only
the field data needs to be repopulated. This is demonstrated with the following pseudo-code which
loops over some parameter `x`:

.. code-block:: python
    :emphasize-lines: 0

    builder = kwant.Builder()
    ...  # specify structural parameters
    system = builder.finalized()

    for x in xs:
        smatrix = kwant.smatrix(system, args=[x])  # apply fields
        ...  # call smatrix methods

This separation is not required with pybinding. As pointed out in the :doc:`/benchmarks/index`,
the fast builder makes it possible to fully reconstruct the model in every loop iteration at no
extra performance cost. This simplifies the code since all the parameters can be applied in a
single place:

.. code-block:: python
    :emphasize-lines: 0

    def make_model(x):
        return pb.Model(..., x)  # all parameters in one place

    for x in xs:
        smatrix = kwant.smatrix(make_model(x).tokwant())  # constructed all at once
        ...  # call smatrix methods

You can :download:`download <kwant_example.py>` a full example file which implements transport
through a barrier like the one presented above. The script uses both builders so you can compare
the implementation as well as the performance. Download the example file and try it on your system.
Our results are presented below (measured using Intel Core i7-4960HQ CPU, 16 GiB RAM, Python 3.5,
macOS 10.11). The size of the square scattering region is increased and we measure the total time
required to calculate the transmission:

.. plot::
    :context: close-figs
    :include-source: False

    sizes = [5, 10, 15, 20, 25, 30]
    pb_times = [2.374, 6.601, 16.207, 31.400, 64.519, 104.874]
    kwant_times = [3.054, 9.781, 22.833, 46.033, 91.581, 146.218]

    plt.figure(figsize=(3, 2.4))
    pb.pltutils.set_palette("Set1", start=3)
    plt.plot(sizes, pb_times, label="pybinding", marker='o', markersize=5, lw=2, zorder=20)
    plt.plot(sizes, kwant_times, label="kwant", marker='o', markersize=5, lw=2, zorder=10)
    plt.grid(True, which='major')
    plt.title("transmission calculation time")
    plt.xlabel("system size (nm)")
    plt.ylabel("time (seconds)")
    plt.xlim(0.8 * min(sizes), 1.05 * max(sizes))
    pb.pltutils.despine()
    pb.pltutils.legend(loc='upper left', reverse=True)

For each system size, the transmission is calculated as a function of barrier height for 100
values. Even though pybinding reconstructs the entire model every time the barrier is changed, the
system build time is so fast that it doesn't affect the total calculation time. In fact, the
extremely fast build actually enables pybinding to outperform Kwant in the overall
calculation. Even though Kwant only repopulates field data at each loop iteration, this still
takes more time than it does for pybinding to fully reconstruct the system.

Note that this example presents a relatively simple system with a square barrier. This is done
to keep the run time to only a few minutes, for convenience. Here, pybinding speeds up the
overall calculation by about 40%. For more realistic examples with larger scattering regions and
complicated field functions with multiple parameters, a speedup of 3-4 times can be achieved by
using pybinding's model builder.


Floating-point precision
------------------------

Pybinding can generate the Hamiltonian matrix with one of four data types: real or complex numbers
with single or double precision (32-bit or 64-bit floating point). The selection is dynamic. The
starting case is always real with single precision and from there the data type is automatically
promoted as needed by the model. For example, adding translationally symmetry or a magnetic field
will cause the builder to switch to complex numbers -- this is detected automatically. On the other
hand, the switch to double precision needs to be requested by the user. The onsite and hopping
energy :ref:`modifiers <modifiers_api>` have an optional `is_double` parameter which can be set to
`True`. The builder switches to double precision if requested by at least one modifier.
Alternatively, :func:`.force_double_precision` can be given to a :class:`.Model` as a direct
parameter.

The reason for all of this is performance. Most solvers work faster with smaller data types: they
consume less memory and bandwidth and SIMD vectorization becomes more efficient. This is assuming
that single precision and/or real numbers are sufficient to describe the given model. In case of
Kwant's solvers, it seems to require double precision in most cases. This is the reason for the
`is_double=True` flag in the above example. Keep this in mind when exporting to Kwant.
