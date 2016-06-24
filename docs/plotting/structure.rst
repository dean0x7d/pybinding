Structure
=========

A structure plot presents the crystal structure of a model by drawing lattice sites as circles and
hoppings as lines which connect the circles. At first glance, this seems like a combination of the
standard scatter and line plots found in matplotlib, but the specific requirements of tight-binding
complicate the implementation. This is why pybinding has its own specialized structure plotting
functions. While these functions are based on matplotlib, they offer additional options which will
be explained here.


Draw only certain hoppings
--------------------------

The system structure plot usually draws lines for all hoppings. We can see an example here with
the third-nearest-neighbor model of graphene. Note the huge number of hoppings in the figure below.
The extra information may be useful for calculations, but it is not always desirable for figures
because of the extra noise. To filter out some of the lines, we can pass the `draw_only` argument
as a list of hopping names. For example, if we only want the first-nearest neighbors:

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    plt.figure(figsize=(8, 3))
    model = pb.Model(graphene.monolayer(nearest_neighbors=3), graphene.hexagon_ac(1))

    plt.subplot(121, title="Unfiltered: all 3 hoppings")
    model.plot()

    plt.subplot(122, title="Filtered: shows only nearest")
    model.plot(hopping={'draw_only': ['t']})

We can also select hoppings in any combination:

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))

    plt.subplot(121, title="$t$ and $t_{nn}$")
    model.plot(hopping={'draw_only': ['t', 't_nn']})

    plt.subplot(122, title="$t$ and $t_{nnn}$")
    model.plot(hopping={'draw_only': ['t', 't_nnn']})


Redraw all axes spines
----------------------

By default, pybinding plots will remove the right and top axes spines. To recover those lines
call the :func:`.pltutils.respine` function.

.. plot::
    :context: close-figs

    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(1))
    model.plot()
    pb.pltutils.respine()


Site radius and color
---------------------

The site radius is given in data units (nanometers in this example). Colors are passed as a list
of colors or a matplotlib colormap.

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(0.5))

    plt.subplot(121, title="Default")
    model.plot()

    plt.subplot(122, title="Customized")
    model.plot(site={'radius': 0.04, 'cmap': ['blue', 'red']})


Hopping width and color
-----------------------

By default, all hopping kinds (nearest, next-nearest, etc.) are shown using the same line color,
but they can be colorized using the `cmap` parameter.

.. plot::
    :context: close-figs

    plt.figure(figsize=(8, 3))
    model = pb.Model(graphene.monolayer(nearest_neighbors=3), pb.rectangle(0.6))

    plt.subplot(121, title="Default")
    model.plot()

    plt.subplot(122, title="Customized")
    model.plot(hopping={'width': 2, 'cmap': 'auto'})
