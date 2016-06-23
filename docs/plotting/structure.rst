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
the third-nearest-neighbor model of graphene:

.. plot::
    :context: close-figs

    from pybinding.repository import graphene

    model = pb.Model(graphene.monolayer(nearest_neighbors=3), graphene.hexagon_ac(1))
    model.system.plot()
    plt.title("Unfiltered: all 3 hoppings")

Note the huge number of hoppings. The extra information may be useful for calculations, but it is
not always desirable for illustration figures because of the extra noise. To filter out some of the
hopping lines, we can pass the `draw_only` argument as a list of hopping names. For example,
if we only want the first-nearest neighbors:

.. plot::
    :context: close-figs

    model.system.plot(hopping={'draw_only': ['t']})
    plt.title("Filtered: shows only nearest")

We can also select hopping in any combination. Here are the first- and third-nearest neighbors:

.. plot::
    :context: close-figs

    model.system.plot(hopping={'draw_only': ['t', 't_nnn']})
    plt.title("Filtered: $t$ and $t_{nnn}$")


Redraw all axes spines
----------------------

By default, pybinding plots will remove the right and top axes spines. To recover those lines
call the :func:`.pltutils.respine` function.

.. plot::
    :context: close-figs

    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(1))
    model.system.plot()
    pb.pltutils.respine()
