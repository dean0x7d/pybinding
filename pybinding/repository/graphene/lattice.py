from math import sqrt

import pybinding as pb
from .constants import *

__all__ = ['monolayer', 'monolayer_alt', 'monolayer_4atom', 'monolayer_nn', 'bilayer']


def monolayer(onsite_a=0, onsite_b=0):
    """Nearest-neighbor monolayer graphene lattice

    Parameters
    ----------
    onsite_a, onsite_b : float
        Onsite energy for sublattices A and B.
    """
    lat = pb.Lattice(a1=[a, 0], a2=[a/2, a/2 * sqrt(3)])

    lat.add_sublattices(
        ('A', [0, -a_cc/2], onsite_a),
        ('B', [0,  a_cc/2], onsite_b)
    )

    lat.add_hoppings(
        ([0,  0], 'A', 'B', t),
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )

    lat.min_neighbors = 2
    return lat


def monolayer_alt(onsite_a=0, onsite_b=0):
    """Nearest-neighbor lattice with alternative lattice vectors

    Parameters
    ----------
    onsite_a, onsite_b : float
        Onsite energy for sublattices A and B.
    """

    lat = pb.Lattice(
        a1=[ a/2, a/2 * sqrt(3)],
        a2=[-a/2, a/2 * sqrt(3)],
    )

    lat.add_sublattices(
        ('A', [0,    0], onsite_a),
        ('B', [0, a_cc], onsite_b)
    )

    # matrix hopping specification
    r0 = [ 0,  0]
    r1 = [ 0, -1]
    r2 = [-1,  0]

    tr0 = [[0, t],
           [t, 0]]
    tr1 = [[0, t],
           [0, 0]]
    tr2 = [[0, t],
           [0, 0]]

    lat.add_hopping_matrices([r0, tr0], [r1, tr1], [r2, tr2])
    lat.min_neighbors = 2
    return lat


def monolayer_4atom(onsite_a=0, onsite_b=0):
    """Nearest-neighbor with 4 atoms per unit cell: square lattice instead of triangular

    Parameters
    ----------
    onsite_a, onsite_b : float
        Onsite energy for sublattices A and B.
    """
    lat = pb.Lattice(a1=[a, 0], a2=[0, 3*a_cc])

    lat.add_sublattices(
        ('A',  [  0, -a_cc/2], onsite_a),
        ('B',  [  0,  a_cc/2], onsite_b),
        ('A2', [a/2,    a_cc], onsite_a, 'A'),
        ('B2', [a/2,  2*a_cc], onsite_b, 'B')
    )

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


def monolayer_nn(onsite_a=0, onsite_b=0, t_nn=0.1):
    """Next-nearest neighbor monolayer lattice

    Parameters
    ----------
    onsite_a, onsite_b : float
        Onsite energy for sublattices A and B.
    t_nn : float
        Next-nearest hopping energy.
    """
    lat = pb.Lattice(a1=[a, 0], a2=[a/2, a/2 * sqrt(3)])

    lat.add_sublattices(
        ('A', [0, -a_cc/2], onsite_a),
        ('B', [0,  a_cc/2], onsite_b)
    )

    lat.add_hoppings(
        # nearest
        ([0,  0], 'A', 'B', t),
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t),
        # next-nearest
        ([0, -1], 'A', 'A', t_nn),
        ([0, -1], 'B', 'B', t_nn),
        ([1, -1], 'A', 'A', t_nn),
        ([1, -1], 'B', 'B', t_nn),
        ([1,  0], 'A', 'A', t_nn),
        ([1,  0], 'B', 'B', t_nn),
    )

    lat.min_neighbors = 2
    return lat


def bilayer(gammas=(), onsite=(0, 0, 0, 0)):
    """Bilayer lattice with optional :math:`\gamma_3` and :math:`\gamma_4` hoppings

    Parameters
    ----------
    gammas : tuple
        By default, only the :math:`\gamma_1` interlayer hopping is used. One or both
        :math:`\gamma_3` and :math:`\gamma_4` can be added with `gammas=(3,)`,
        `gammas=(4,)` or `gammas=(3, 4)`.
    onsite : tuple
        Onsite energy for A1, B1, A2, B2
    """
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

    if 3 in gammas:
        lat.add_hoppings(
            ([0, 1], 'B2', 'A1', 'gamma3'),
            ([1, 0], 'B2', 'A1', 'gamma3'),
            ([1, 1], 'B2', 'A1', 'gamma3')
        )

    if 4 in gammas:
        lat.add_hoppings(
            ([0, 0], 'A2', 'A1', 'gamma4'),
            ([0, 1], 'A2', 'A1', 'gamma4'),
            ([1, 0], 'A2', 'A1', 'gamma4')
        )

    lat.min_neighbors = 2
    return lat
