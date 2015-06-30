import math

import pybinding as pb
from .constants import *


def monolayer(onsite_a=0, onsite_b=0):
    lat = pb.Lattice([a, 0], [0.5 * a, 0.5 * math.sqrt(3) * a])

    lat.add_sublattices(
        ['a', (0, -a_cc/2), onsite_a],
        ['b', (0,  a_cc/2), onsite_b]
    )

    # sparse hopping specification
    lat.add_hoppings(
        [(0,  0), 'a', 'b', t],
        [(1, -1), 'a', 'b', t],
        [(0, -1), 'a', 'b', t]
    )

    lat.min_neighbors = 2
    return lat


def monolayer_alt(onsite_a=0, onsite_b=0):
    """ Alternative graphene lattice specification: different lattice vectors """
    lat = pb.Lattice(
        [ 0.5 * a, 0.5 * math.sqrt(3) * a],
        [-0.5 * a, 0.5 * math.sqrt(3) * a],
    )

    lat.add_sublattices(
        ['a', (0,    0), onsite_a],
        ['b', (0, a_cc), onsite_b]
    )

    # matrix hopping specification
    r0 = ( 0,  0)
    r1 = ( 0, -1)
    r2 = (-1,  0)

    tr0 = [(0, t),
           (t, 0)]
    tr1 = [(0, t),
           (0, 0)]
    tr2 = [(0, t),
           (0, 0)]

    lat.add_hopping_matrices([r0, tr0], [r1, tr1], [r2, tr2])
    lat.min_neighbors = 2
    return lat


def monolayer_4atom(onsite_a=0, onsite_b=0):
    """ Graphene with 4 atoms per unit cell: square lattice instead of triangular """
    lat = pb.Lattice([a, 0], [0, 3*a_cc])

    lat.add_sublattices(
        ['a', (-a/4, -a_cc * 5/4), onsite_a],
        ['b', (-a/4,     -a_cc/4), onsite_b],
        ['a2', (a/4,      a_cc/4), onsite_a, 'a'],
        ['b2', (a/4,  a_cc * 5/4), onsite_b, 'b']
    )

    lat.add_hoppings(
        # inside the unit sell
        [(0, 0), 'a',  'b',  t],
        [(0, 0), 'b',  'a2', t],
        [(0, 0), 'a2', 'b2', t],
        # between neighbouring unit cells
        [(-1, -1), 'a', 'b2', t],
        [( 0, -1), 'a', 'b2', t],
        [(-1,  0), 'b', 'a2', t],
    )

    lat.min_neighbors = 2
    return lat


def monolayer_nn(onsite_a=0, onsite_b=0):
    """ Next-nearest neighbour model of graphene """
    lat = pb.Lattice([a, 0], [0.5 * a, 0.5*math.sqrt(3) * a])

    lat.add_sublattices(
        ['a', (0, -a_cc/2), onsite_a],
        ['b', (0,  a_cc/2), onsite_b]
    )

    lat.add_hoppings(
        # nearest
        [(0,  0), 'a', 'b', t],
        [(1, -1), 'a', 'b', t],
        [(0, -1), 'a', 'b', t],
        # next-nearest
        [(0, -1), 'a', 'a', t_nn],
        [(0, -1), 'b', 'b', t_nn],
        [(1, -1), 'a', 'a', t_nn],
        [(1, -1), 'b', 'b', t_nn],
        [(1,  0), 'a', 'a', t_nn],
        [(1,  0), 'b', 'b', t_nn],
    )

    lat.min_neighbors = 2
    return lat


def bilayer(onsite_a1=0, onsite_b1=0, onsite_a2=0, onsite_b2=0):
    c0 = 0.335  # [nm] interlayer spacing
    gamma1 = -0.4
    # gamma3 = -0.3
    # gamma4 = -0.04

    lat = pb.Lattice(
        [ 0.5 * a, 0.5 * math.sqrt(3) * a],
        [-0.5 * a, 0.5 * math.sqrt(3) * a]
    )

    lat.add_sublattices(
        ['a1', (0,  -a_cc/2,   0), onsite_a1],
        ['b1', (0,   a_cc/2,   0), onsite_b1],
        ['a2', (0,   a_cc/2, -c0), onsite_a2],
        ['b2', (0, 3*a_cc/2, -c0), onsite_b2]
    )

    lat.add_hoppings(
        [( 0,  0), 'a1', 'b1', t],
        [( 0, -1), 'a1', 'b1', t],
        [(-1,  0), 'a1', 'b1', t],
        [( 0,  0), 'a2', 'b2', t],
        [( 0, -1), 'a2', 'b2', t],
        [(-1,  0), 'a2', 'b2', t],
        [( 0,  0), 'b1', 'a2', gamma1],
        # [(0, 1), 'b2', 'a1', gamma3],
        # [(1, 0), 'b2', 'a1', gamma3],
        # [(1, 1), 'b2', 'a1', gamma3],
        # [(0, 0), 'a2', 'a1', gamma4],
        # [(0, 1), 'a2', 'a1', gamma4],
        # [(1, 0), 'a2', 'a1', gamma4],
    )

    lat.min_neighbors = 2
    return lat
