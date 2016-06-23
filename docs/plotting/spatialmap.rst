Spatial map
===========

TODO

Draw only certain hoppings
--------------------------

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    model = pb.Model(graphene.monolayer(nearest_neighbors=3), graphene.hexagon_ac(1))
    model.plot(hopping={'draw_only': ['t']})


.. plot::
    :context: close-figs

    solver = pb.solver.arpack(model, k=10)
    smap = solver.calc_probability(n=2)
    smap.plot_structure(hopping={'draw_only': ['t']})
    pb.pltutils.colorbar(label="$|\Psi|^2$")


Slicing
-------

.. plot::
    :context: close-figs

    smap.filter(smap.pos.y > 0)
    smap.plot_structure(hopping={'draw_only': ['t']})
    pb.pltutils.colorbar(label="$|\Psi|^2$")
