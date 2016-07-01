Structure-mapped data
=====================

As shown in the previous section, many classes in Pybinding use structure plots in a similar way.
One class stands out here: :class:`.StructureMap` can be used to map any arbitrary data onto the
spatial structure of a model. :class:`.StructureMap` objects are produced in two cases: as the
results of various computation functions (e.g. :meth:`.Solver.calc_spatial_ldos`) or returned
from :meth:`.Model.structure_map` which can map custom user data.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


Draw only certain hoppings
--------------------------

Just as before, we can draw only the desired hoppings. Note that `smap` is a :class:`.StructureMap`
returned by :meth:`.Solver.calc_probability`.

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
    smap.plot(hopping={'draw_only': ['t']})
    pb.pltutils.colorbar()


Slicing a structure
-------------------

This follows a syntax similar to numpy fancy indexing where we can give a condition as the index.

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="Original")
    smap.plot(hopping={'draw_only': ['t']})

    plt.subplot(122, title="Sliced: y > 0")
    upper = smap[smap.y > 0]
    upper.plot(hopping={'draw_only': ['t']})


.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="Original: A and B")
    smap.plot(hopping={'draw_only': ['t', 't_nn']})

    plt.subplot(122, title="Sliced: A only")
    a_only = smap[smap.sublattices == 'A']
    a_only.plot(hopping={'draw_only': ['t', 't_nn']})


Mapping custom data
-------------------

The method :meth:`.Model.structure_map` returns a :class:`.StructureMap` where any user-defined
`data` can be mapped to the spatial positions of the lattice sites. The `data` just needs to be
a 1D array with the same size as the total number of sites in the system.

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="The model")
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(1))
    model.plot()

    plt.subplot(122, title="Custom color data")
    custom_data = np.linspace(0, 2, num=model.system.num_sites)
    smap = model.structure_map(custom_data)
    smap.plot()
    pb.pltutils.colorbar()


.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="sin(10x)")
    smap = model.structure_map(np.sin(10 * model.system.x))
    smap.plot()
    pb.pltutils.colorbar()

    plt.subplot(122, title="cos(5y)")
    smap = model.structure_map(np.cos(5 * model.system.y))
    smap.plot()
    pb.pltutils.colorbar()


Contour plots for large systems
-------------------------------

For larger systems, structure plots don't make much sense because the details of the sites and
hoppings would be too small to see. Contour plots look much better in this case.

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10))

    plt.subplot(121, title="sin(x)")
    smap = model.structure_map(np.sin(model.system.x))
    smap.plot_contourf()
    pb.pltutils.colorbar()

    plt.subplot(122, title="cos(y/2)")
    smap = model.structure_map(np.cos(0.5 * model.system.y))
    smap.plot_contourf()
    pb.pltutils.colorbar()
