import math

from pybinding.lattice import Lattice
from .constants import *


def monolayer(onsite_a=0, onsite_b=0):
    lat = Lattice(min_neighbors=2)
    lat.set_vectors([a, 0], [0.5 * a, 0.5 * math.sqrt(3) * a])

    sub_a = lat.create_sublattice((0, -a_cc/2), onsite_a, name='a')
    sub_b = lat.create_sublattice((0,  a_cc/2), onsite_b, name='b')

    # sparse hopping specification
    lat.set_hoppings(
        [(0,  0), sub_a, sub_b, t],
        [(1, -1), sub_a, sub_b, t],
        [(0, -1), sub_a, sub_b, t]
    )
    return lat


def monolayer_alt(onsite_a=0, onsite_b=0):
    """ Alternative graphene lattice specification: different lattice vectors """
    lat = Lattice(min_neighbors=2)
    lat.set_vectors(
        [ 0.5 * a, 0.5 * math.sqrt(3) * a],
        [-0.5 * a, 0.5 * math.sqrt(3) * a],
    )

    lat.create_sublattice((0,    0), onsite_a, name='a')
    lat.create_sublattice((0, a_cc), onsite_b, name='b')

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

    lat.set_hopping_matrix([r0, tr0], [r1, tr1], [r2, tr2])
    return lat


def monolayer_4atom(onsite_a=0, onsite_b=0):
    """ Graphene with 4 atoms per unit cell: square lattice instead of triangular """
    lat = Lattice(min_neighbors=2)
    lat.set_vectors([a, 0], [0, 3*a_cc])

    a1 = lat.create_sublattice((-a/4, -a_cc * 5/4), onsite_a, name='a')
    b1 = lat.create_sublattice((-a/4,     -a_cc/4), onsite_b, name='b')
    a2 = lat.create_sublattice(( a/4,      a_cc/4), onsite_a, alias=a1)
    b2 = lat.create_sublattice(( a/4,  a_cc * 5/4), onsite_b, alias=b1)

    lat.set_hoppings(
        # inside the unit sell
        [( 0,  0), a1, b1, t],
        [( 0,  0), b1, a2, t],
        [( 0,  0), a2, b2, t],
        # between neighbouring unit cells
        [(-1, -1), a1, b2, t],
        [( 0, -1), a1, b2, t],
        [(-1,  0), b1, a2, t],
    )
    return lat


def monolayer_nn(onsite_a=0, onsite_b=0):
    """ Next-nearest neighbour model of graphene """
    lat = Lattice(min_neighbors=2)
    lat.set_vectors([a, 0], [0.5 * a, 0.5*math.sqrt(3) * a])

    sub_a = lat.create_sublattice((0, -a_cc/2), onsite_a, name='a')
    sub_b = lat.create_sublattice((0,  a_cc/2), onsite_b, name='b')

    lat.set_hoppings(
        # nearest
        [( 0,  0), sub_a, sub_b, t],
        [( 1, -1), sub_a, sub_b, t],
        [( 0, -1), sub_a, sub_b, t],
        # next-nearest
        [( 0, -1), sub_a, sub_a, t_nn],
        [( 0, -1), sub_b, sub_b, t_nn],
        [( 1, -1), sub_a, sub_a, t_nn],
        [( 1, -1), sub_b, sub_b, t_nn],
        [( 1,  0), sub_a, sub_a, t_nn],
        [( 1,  0), sub_b, sub_b, t_nn],
    )
    return lat


def bilayer(onsite_a1=0, onsite_b1=0, onsite_a2=0, onsite_b2=0):
    c0 = 0.335  # [nm] interlayer spacing
    gamma1 = -0.4
    # gamma3 = -0.3
    # gamma4 = -0.04

    lat = Lattice(min_neighbors=2)
    lat.set_vectors(
        [ 0.5 * a, 0.5 * math.sqrt(3) * a],
        [-0.5 * a, 0.5 * math.sqrt(3) * a],
    )

    a1 = lat.create_sublattice((0,  -a_cc/2,   0), onsite_a1, name='a1')
    b1 = lat.create_sublattice((0,   a_cc/2,   0), onsite_b1, name='b1')
    a2 = lat.create_sublattice((0,   a_cc/2, -c0), onsite_a2, name='a2')
    b2 = lat.create_sublattice((0, 3*a_cc/2, -c0), onsite_b2, name='b2')

    lat.set_hoppings(
        [( 0,  0),  a1, b1, t],
        [( 0, -1),  a1, b1, t],
        [(-1,  0),  a1, b1, t],
        [( 0,  0),  a2, b2, t],
        [( 0, -1),  a2, b2, t],
        [(-1,  0),  a2, b2, t],
        [( 0,  0),  b1, a2, gamma1],
        # [( 0,  1),  b2, a1, gamma3],
        # [( 1,  0),  b2, a1, gamma3],
        # [( 1,  1),  b2, a1, gamma3],
        # [( 0,  0),  a2, a1, gamma4],
        # [( 0,  1),  a2, a1, gamma4],
        # [( 1,  0),  a2, a1, gamma4],
    )
    return lat
