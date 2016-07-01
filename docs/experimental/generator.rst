Hopping generator
=================

The :func:`@hopping_generator <.hopping_generator>` can be used to create new hoppings independent
of the main lattice definition. It’s especially useful for creating additional local hoppings,
e.g. to model defects. Here, we present a way create twisted bilayer graphene with an arbitrary
rotation angle :math:`\theta`.

We start with two unconnected layers of graphene. A :func:`@site_position_modifier <.site_position_modifier>`
is applied to rotate just one layer. Then, a :func:`@hopping_generator <.hopping_generator>` finds
and connects the layers via site pairs which satisfy the given criteria. The newly created hoppings
all have identical energy at first. Finally, a :func:`@hopping_energy_modifier <.hopping_energy_modifier>`
to applied to set the new interlayer hopping energy to the desired distance-dependent value.

This is an experimental feature, presented as is, without any additional support.

:download:`Source code </experimental/twisted_bilayer.py>`

.. plot:: experimental/twisted_bilayer.py
    :include-source:
    :alt: Twisted bilayer graphene for arbitrary angles
