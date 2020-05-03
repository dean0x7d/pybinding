Generators
==========

.. meta::
   :description: Generating new hoppings and sites

Up to now, we've only been using **modifiers** in order to manipulate existing lattice features
and their corresponding Hamiltonian matrix terms (:func:`@site_position_modifier <.site_position_modifier>`,
:func:`@site_state_modifier <.site_state_modifier>`, :func:`@onsite_energy_modifier <.onsite_energy_modifier>`,
:func:`@hopping_energy_modifier <.hopping_energy_modifier>`).
This section introduces **generators** which can be used to create completely new sites or hoppings
that are independent of the main :class:`.Lattice` definition.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

A limiting case for the use of different modifiers would be a lattice decorated with adatoms,
or an impurity site that resides on a material surface. These cases can not be covered with
modifiers as neither the sites nor the hopping terms exist in our Hamiltonian. In this case one
needs to generate additional sites and introduce a new family of hopping terms inside the model.
:func:`@hopping_generator <.hopping_generator>` and :func:`@site_generator <.site_generator>`
cover such cases.

The :func:`@hopping_generator <.hopping_generator>` can be used to create arbitrary new hoppings.
It’s especially useful for creating additional local hoppings, e.g. to model defects. Here, we
present a way to create twisted bilayer graphene with an arbitrary rotation angle :math:`\theta`.
We start with two unconnected layers of graphene.

.. plot::
    :nofigs:
    :context:

    import math
    import pybinding as pb

    c0 = 0.335  # [nm] graphene interlayer spacing


    def two_graphene_monolayers():
        """Two individual AB stacked layers of monolayer graphene without interlayer hopping"""
        from pybinding.repository.graphene.constants import a_cc, a, t

        lat = pb.Lattice(a1=[a/2, a/2 * math.sqrt(3)], a2=[-a/2, a/2 * math.sqrt(3)])
        lat.add_sublattices(('A1', [0,   a_cc,   0]),
                            ('B1', [0,      0,   0]),
                            ('A2', [0,      0, -c0]),
                            ('B2', [0,  -a_cc, -c0]))
        lat.register_hopping_energies({'gamma0': t})
        lat.add_hoppings(
            # layer 1
            ([0, 0], 'A1', 'B1', 'gamma0'),
            ([0, 1], 'A1', 'B1', 'gamma0'),
            ([1, 0], 'A1', 'B1', 'gamma0'),
            # layer 2
            ([0, 0], 'A2', 'B2', 'gamma0'),
            ([0, 1], 'A2', 'B2', 'gamma0'),
            ([1, 0], 'A2', 'B2', 'gamma0'),
            # not interlayer hopping
        )
        lat.min_neighbors = 2
        return lat

A :func:`@site_position_modifier <.site_position_modifier>` is applied to rotate just one layer.
Then, a :func:`@hopping_generator <.hopping_generator>` finds and connects the layers via site
pairs which satisfy the given criteria using :class:`~scipy.spatial.cKDTree`.

.. plot::
    :nofigs:
    :context:

    from scipy.spatial import cKDTree

    def twist_layers(theta):
        """Rotate one layer and then a generate hopping between the rotated layers,
           reference is AB stacked"""
        theta = theta / 180 * math.pi  # from degrees to radians

        @pb.site_position_modifier
        def rotate(x, y, z):
            """Rotate layer 2 by the given angle `theta`"""
            layer2 = (z < 0)
            x0 = x[layer2]
            y0 = y[layer2]
            x[layer2] = x0 * math.cos(theta) - y0 * math.sin(theta)
            y[layer2] = y0 * math.cos(theta) + x0 * math.sin(theta)
            return x, y, z

        @pb.hopping_generator('interlayer', energy=0.1)  # eV
        def interlayer_generator(x, y, z):
            """Generate hoppings for site pairs which have distance `d_min < d < d_max`"""
            positions = np.stack([x, y, z], axis=1)
            layer1 = (z == 0)
            layer2 = (z != 0)

            d_min = c0 * 0.98
            d_max = c0 * 1.1
            kdtree1 = cKDTree(positions[layer1])
            kdtree2 = cKDTree(positions[layer2])
            coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

            idx = coo.data > d_min
            abs_idx1 = np.flatnonzero(layer1)
            abs_idx2 = np.flatnonzero(layer2)
            row, col = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]
            return row, col  # lists of site indices to connect

        @pb.hopping_energy_modifier
        def interlayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
            """Set the value of the newly generated hoppings as a function of distance"""
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            interlayer = (hop_id == 'interlayer')
            energy[interlayer] = 0.4 * c0 / d[interlayer]
            return energy

        return rotate, interlayer_generator, interlayer_hopping_value

The newly created hoppings all have identical energy at first. Finally,
a :func:`@hopping_energy_modifier <.hopping_energy_modifier>` is applied to set the new interlayer
hopping energy to the desired distance-dependent value.

.. plot::
    :context: close-figs
    :alt: Twisted bilayer graphene

    model = pb.Model(
        two_graphene_monolayers(),
        pb.circle(radius=1.5),
        twist_layers(theta=21.798)
    )
    plt.figure(figsize=(6.5, 6.5))
    model.plot()
    plt.title(r"$\theta$ = 21.798 $\degree$")

This example covers a structure with two equivalent layers, both of which are defined using a
:class:`.Lattice`. A similar approach can be used when considering heterostructures but we can use
a :func:`@site_generator <.site_generator>` to add a layer created from a different unit cell.

.. plot::
    :nofigs:
    :context:

    def hbn_layer(shape):
        """Generate hBN layer defined by the shape with intralayer hopping terms"""
        from pybinding.repository.graphene.constants import a_cc, t

        a_bn = 56 / 55 * a_cc  # ratio of lattice constants is 56/55

        vn = -1.4  # [eV] nitrogen onsite potential
        vb = 3.34  # [eV] boron onsite potential

        def hbn_monolayer():
            """Create a lattice of monolayer hBN """

            a = math.sqrt(3) * a_bn
            lat = pb.Lattice(a1=[a/2, a/2 * math.sqrt(3)], a2=[-a/2, a/2 * math.sqrt(3)])
            lat.add_sublattices(('Br', [0, -a_bn,   -c0], vb),
                                ('N', [0,     0,   -c0], vn))

            lat.min_neighbors = 2  # no need for hoppings lattice is used only to generate coordinates
            return lat

        model = pb.Model(
            hbn_monolayer(),
            shape
        )

        subs = model.system.sublattices
        idx_b = np.flatnonzero(subs == model.lattice.sublattices["Br"].alias_id)
        idx_n = np.flatnonzero(subs == model.lattice.sublattices["N"].alias_id)
        positions_boron    = model.system[idx_b].positions
        positions_nitrogen = model.system[idx_n].positions

        @pb.site_generator(name='Br', energy=vb)  # onsite energy [eV]
        def add_boron():
            """Add positions of newly generated boron sites"""
            return positions_boron

        @pb.site_generator(name='N', energy=vn)  # onsite energy [eV]
        def add_nitrogen():
            """Add positions of newly generated nitrogen sites"""
            return positions_nitrogen

        @pb.hopping_generator('intralayer_bn', energy=t)
        def intralayer_generator(x, y, z):
            """Generate nearest-neighbor hoppings between B and N sites"""
            positions = np.stack([x, y, z], axis=1)
            layer_bn = (z != 0)

            d_min = a_bn * 0.98
            d_max = a_bn * 1.1
            kdtree1 = cKDTree(positions[layer_bn])
            kdtree2 = cKDTree(positions[layer_bn])
            coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

            idx = coo.data > d_min
            abs_idx = np.flatnonzero(layer_bn)

            row, col = abs_idx[coo.row[idx]], abs_idx[coo.col[idx]]
            return row, col  # lists of site indices to connect

        @pb.hopping_energy_modifier
        def intralayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
            """Set the value of the newly generated hoppings as a function of distance"""
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            intralayer = (hop_id == 'intralayer_bn')
            energy[intralayer] = 0.1 * t * a_bn / d[intralayer]
            return energy

        return add_boron, add_nitrogen, intralayer_generator, intralayer_hopping_value

Function :func:`hbn_layer` creates a layer of
hexagonal boron-nitride that fits a given shape, and connects the intralayer sites, while :func:`graphene.monolayer_alt()
<.graphene.lattice.monolayer_alt>` creates a single layer of graphene. We can once again use the function
:func:`twist_layers` and create the desired graphene/hBN flake.

.. plot::
    :context: close-figs
    :alt: Graphene/hexagonal boron-nitride

    from pybinding.repository import graphene

    shape = pb.circle(radius=2),

    model = pb.Model(
        graphene.monolayer_alt(),  # reference stacking is AB (theta=0)
        shape,
        hbn_layer(shape=shape),
        twist_layers(21.787),
    )
    plt.figure(figsize=(6.8, 7.5))
    s = model.system
    plt.subplot(2, 2, 1, title="graphene")
    s[s.z == 0].plot()
    plt.subplot(2, 2, 2, title="hBN")
    s[s.z < 0].plot()
    plt.subplot(2, 2, (3, 4), title="graphene/hBN")
    s.plot()

.. note::
    Site and hopping generators are applied at an earlier stage of a model construction and the
    order in which are passed to :class:`.Model` matters. To be more precise, although modifiers
    can be freely ordered between themselves, the ordering of modifiers with respect to generators
    may affect the final model.

A similar approach for creating a heterostructure can rely of incorporating all moiré sites within
the :class:`.Lattice` object. In that case, periodic boundary conditions could be applied in a
straightforward way, which, for example, allows the computation of the band structure. Take into
account that a hopping check is performed each time a new hopping term is added to the lattice/model,
which would increase the model constructions time for lattices exceeding millions of hoppings.
Finally, it is up to the user to chose an approach which suits their needs better.


Further reading
---------------

Take a look at the :ref:`generators_api` API reference for more information.


Examples
--------

:download:`Source code </tutorial/twisted_heterostructures.py>`

.. plot:: tutorial/twisted_heterostructures.py
    :include-source:
    :alt: Twisted bilayer graphene and graphene/hBN flakes for arbitrary angles
