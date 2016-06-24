Spatial map
===========

TODO

Draw only certain hoppings
--------------------------

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="The model")
    model = pb.Model(graphene.monolayer(nearest_neighbors=3), graphene.hexagon_ac(1))
    model.plot(hopping={'draw_only': ['t']})

    plt.subplot(122, title="$|\Psi|^2$")
    solver = pb.solver.arpack(model, k=10)
    smap = solver.calc_probability(n=2)
    smap.plot_structure(hopping={'draw_only': ['t']})
    pb.pltutils.colorbar()


Slicing
-------

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="Original")
    smap.plot_structure(hopping={'draw_only': ['t']})

    plt.subplot(122, title="Sliced: y > 0")
    upper = smap[smap.y > 0]
    upper.plot_structure(hopping={'draw_only': ['t']})


.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="Original: A and B")
    smap.plot_structure(hopping={'draw_only': ['t', 't_nn']})

    plt.subplot(122, title="Sliced: A only")
    a_only = smap[smap.sublattices == 'A']
    a_only.plot_structure(hopping={'draw_only': ['t', 't_nn']})
