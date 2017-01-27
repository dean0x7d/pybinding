import pybinding as pb

__all__ = ['monolayer', 'monolayer_alt', 'monolayer_4atom', 'bilayer']


def monolayer(nearest_neighbors=1, onsite=(0, 0), **kwargs):
    """Monolayer graphene lattice up to `nearest_neighbors` hoppings

    Parameters
    ----------
    nearest_neighbors : int
        Number of nearest neighbors to consider.
    onsite : Tuple[float, float]
        Onsite energy for sublattices A and B.
    **kwargs
        Specify the hopping parameters `t`, `t_nn` and `t_nnn`.
        If not given, the default values from :mod:`.graphene.constants` will be used.
    """
    from math import sqrt
    from .constants import a_cc, a, t, t_nn

    lat = pb.Lattice(a1=[a, 0], a2=[a/2, a/2 * sqrt(3)])

    # The next-nearest hoppings shift the Dirac point away from zero energy.
    # This will push it back to zero for consistency with the first-nearest model.
    onsite_offset = 0 if nearest_neighbors < 2 else 3 * kwargs.get('t_nn', t_nn)

    lat.add_sublattices(
        ('A', [0, -a_cc/2], onsite[0] + onsite_offset),
        ('B', [0,  a_cc/2], onsite[1] + onsite_offset)
    )

    lat.register_hopping_energies({
        't': kwargs.get('t', t),
        't_nn': kwargs.get('t_nn', t_nn),
        't_nnn': kwargs.get('t_nnn', 0.05),
    })

    lat.add_hoppings(
        ([0,  0], 'A', 'B', 't'),
        ([1, -1], 'A', 'B', 't'),
        ([0, -1], 'A', 'B', 't')
    )

    if nearest_neighbors >= 2:
        lat.add_hoppings(
            ([0, -1], 'A', 'A', 't_nn'),
            ([0, -1], 'B', 'B', 't_nn'),
            ([1, -1], 'A', 'A', 't_nn'),
            ([1, -1], 'B', 'B', 't_nn'),
            ([1,  0], 'A', 'A', 't_nn'),
            ([1,  0], 'B', 'B', 't_nn'),
        )

    if nearest_neighbors >= 3:
        lat.add_hoppings(
            [( 1, -2), 'A', 'B', 't_nnn'],
            [( 1,  0), 'A', 'B', 't_nnn'],
            [(-1,  0), 'A', 'B', 't_nnn'],
        )

    if nearest_neighbors >= 4:
        raise RuntimeError("No more")

    lat.min_neighbors = 2
    return lat


def monolayer_alt(onsite=(0, 0)):
    """Nearest-neighbor lattice with alternative lattice vectors

    Parameters
    ----------
    onsite : Tuple[float, float]
        Onsite energy for sublattices A and B.
    """
    from math import sqrt
    from .constants import a_cc, a, t

    lat = pb.Lattice(a1=[a/2,  a/2 * sqrt(3)],
                     a2=[a/2, -a/2 * sqrt(3)])

    lat.add_sublattices(('A', [0,    0], onsite[0]),
                        ('B', [0, a_cc], onsite[1]))

    lat.add_hoppings(([ 0,  0], 'A', 'B', t),
                     ([ 0,  1], 'A', 'B', t),
                     ([-1,  0], 'A', 'B', t))

    lat.min_neighbors = 2
    return lat


def monolayer_4atom(onsite=(0, 0)):
    """Nearest-neighbor with 4 atoms per unit cell: square lattice instead of oblique

    Parameters
    ----------
    onsite : Tuple[float, float]
        Onsite energy for sublattices A and B.
    """
    from .constants import a_cc, a, t

    lat = pb.Lattice(a1=[a, 0], a2=[0, 3*a_cc])

    lat.add_sublattices(('A',  [  0, -a_cc/2], onsite[0]),
                        ('B',  [  0,  a_cc/2], onsite[1]))

    lat.add_aliases(('A2', 'A', [a / 2, a_cc]),
                    ('B2', 'B', [a / 2, 2 * a_cc]))

    lat.add_hoppings(
        # inside the unit sell
        ([0, 0], 'A',  'B',  t),
        ([0, 0], 'B',  'A2', t),
        ([0, 0], 'A2', 'B2', t),
        # between neighbouring unit cells
        ([-1, -1], 'A', 'B2', t),
        ([ 0, -1], 'A', 'B2', t),
        ([-1,  0], 'B', 'A2', t),
    )

    lat.min_neighbors = 2
    return lat


def bilayer(gamma3=False, gamma4=False, onsite=(0, 0, 0, 0)):
    """Bilayer lattice in the AB-stacked form (Bernal-stacked)

    * :math:`\gamma_0` is the single-layer hopping within the top layer (A1/B1)
      and bottom layer (A2/B2)
    * :math:`\gamma_1` is the inter-layer hopping between B1 and A2
      (where atom B1 lies directly over A2)
    * Hoppings :math:`\gamma_3` and :math:`\gamma_4` are optional (see parameters)

    Parameters
    ----------
    gamma3, gamma4 : bool
        Enable :math:`\gamma_3` and/or :math:`\gamma_4` hoppings.
        By default, only :math:`\gamma_0` and :math:`\gamma_1` are active.
    onsite : Tuple[float, float, float, float]
        Onsite energy for A1, B1, A2, B2
    """
    from math import sqrt
    from .constants import a_cc, a, t

    lat = pb.Lattice(
        a1=[ a/2, a/2 * sqrt(3)],
        a2=[-a/2, a/2 * sqrt(3)]
    )

    c0 = 0.335  # [nm] interlayer spacing
    lat.add_sublattices(
        ('A1', [0,  -a_cc/2,   0], onsite[0]),
        ('B1', [0,   a_cc/2,   0], onsite[1]),
        ('A2', [0,   a_cc/2, -c0], onsite[2]),
        ('B2', [0, 3*a_cc/2, -c0], onsite[3])
    )

    lat.register_hopping_energies({
        'gamma0': t,
        'gamma1': -0.4,
        'gamma3': -0.3,
        'gamma4': -0.04
    })

    lat.add_hoppings(
        # layer 1
        ([ 0,  0], 'A1', 'B1', 'gamma0'),
        ([ 0, -1], 'A1', 'B1', 'gamma0'),
        ([-1,  0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([ 0,  0], 'A2', 'B2', 'gamma0'),
        ([ 0, -1], 'A2', 'B2', 'gamma0'),
        ([-1,  0], 'A2', 'B2', 'gamma0'),
        # interlayer
        ([ 0,  0], 'B1', 'A2', 'gamma1')
    )

    if gamma3:
        lat.add_hoppings(
            ([0, 1], 'B2', 'A1', 'gamma3'),
            ([1, 0], 'B2', 'A1', 'gamma3'),
            ([1, 1], 'B2', 'A1', 'gamma3')
        )

    if gamma4:
        lat.add_hoppings(
            ([0, 0], 'A2', 'A1', 'gamma4'),
            ([0, 1], 'A2', 'A1', 'gamma4'),
            ([1, 0], 'A2', 'A1', 'gamma4')
        )

    lat.min_neighbors = 2
    return lat
