Shape and symmetry
------------------

The last two sections showed how to model shape and symmetry individually, but we can be more
creative and combine the two.


Example
*******

Note the zero-energy mode in the band structure. For wave vector :math:`k = 0`, states on the
outer edge of the ring have the highest LDOS intensity, but for :math:`k = \pi / a` the inner
edge states dominate.


:download:`Source code </tutorial/shape_symmetry_example.py>`

.. plot:: tutorial/shape_symmetry_example.py
    :include-source:


Nanoribbons
***********

To create a graphene nanoribbon, we'll need a shape to give the finite width of the ribbon
while the infinite length is achieved with translational symmetry.

.. plot::
    :context: reset

    from pybinding.repository import graphene

    model = pb.Model(
        graphene.lattice.monolayer(),
        pb.rectangle(1.2),  # nm
        pb.translational_symmetry(a1=True, a2=False)
    )
    model.system.plot()
    model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm

As before, the central darker circles represent the main cell of the nanoribbon, the lighter
colored circles are the translations due to symmetry and the red lines are boundary hoppings.
The two arrows in the upper left corner show the primitive lattice vectors of graphene.

The :func:`.translational_symmetry` is applied only in the :math:`a_1` lattice vector direction
which gives the ribbon its infinite length, but the symmetry is disabled in the :math:`a_2`
direction so that the finite size of the shape is preserved. The builtin :func:`.rectangle` shape
gives the nanoribbon its 1.2 nm width.

The band structure calculations work just as before.

.. plot::
    :context: close-figs

    from math import pi, sqrt

    solver = pb.solver.lapack(model)
    a = graphene.a_cc * sqrt(3)  # ribbon unit cell length
    bands = solver.calc_bands(-pi/a, pi/a)
    bands.plot()


.. todo::
    Translations symmetry with `float` arguments

.. todo::
    4 atom unit cell


Complex structures
******************

The combination of shape and symmetry can be more complex as shown here with a nanoribbon ring
structure.

.. plot::
    :context: close-figs

    def ring(inner_radius, outer_radius):
        def contains(x, y, z):
            r = np.sqrt(x**2 + y**2)
            return np.logical_and(inner_radius < r, r < outer_radius)

        return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])


    model = pb.Model(
        graphene.lattice.monolayer_4atom(),
        ring(inner_radius=1.4, outer_radius=2),
        pb.translational_symmetry(a1=3.8, a2=False)
    )

    plt.figure(figsize=pb.pltutils.cm2inch(20, 7))
    model.system.plot()

The period length of the translation in the :math:`a_1` direction is set to 3.8 nm. This ensures
that the inner ring shape is preserved and the periodic boundaries are placed on the outer edges.


.. plot::
    :context: close-figs

    solver = pb.solver.arpack(model, num_eigenvalues=10)
    a = 3.8  # [nm] unit cell length
    bands = solver.calc_bands(-pi/a, pi/a)
    bands.plot(point_labels=['$-\pi / a$', '$\pi / a$'])


.. todo::
    2D periodic structure


Further reading
***************

.. todo::
    For more symmetry tricks check out ...
