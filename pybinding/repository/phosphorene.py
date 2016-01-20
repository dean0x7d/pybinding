"""A single layer of black phosphorus"""
from math import pi, sin, cos
import pybinding as pb


def monolayer_4band():
    """Monolayer phosphorene lattice using the four-band model"""
    a = 0.222
    ax = 0.438
    ay = 0.332
    theta = 96.79 * (pi / 180)
    phi = 103.69 * (pi / 180)

    lat = pb.Lattice(a1=[ax, 0], a2=[0, ay])

    h = a * sin(phi - pi / 2)
    s = 0.5 * ax - a * cos(theta / 2)
    lat.add_sublattices(
        ('A', [0,           0, h], 0),
        ('B', [s,           0, 0], 0),
        ('C', [ax/2,     ay/2, 0], 0),
        ('D', [ax/2 + s, ay/2, h], 0)
    )

    lat.register_hopping_energies({
        't1': -1.22,
        't2': 3.665,
        't3': -0.205,
        't4': -0.105,
        't5': -0.055
    })

    lat.add_hoppings(
        # t1
        ([-1,  0], 'A', 'D', 't1'),
        ([-1, -1], 'A', 'D', 't1'),
        ([ 0,  0], 'B', 'C', 't1'),
        ([ 0, -1], 'B', 'C', 't1'),
        # t2
        ([ 0,  0], 'A', 'B', 't2'),
        ([ 0,  0], 'C', 'D', 't2'),
        # t3
        ([ 0,  0], 'A', 'D', 't3'),
        ([ 0, -1], 'A', 'D', 't3'),
        ([ 1,  1], 'C', 'B', 't3'),
        ([ 1,  0], 'C', 'B', 't3'),
        # t4
        ([ 0,  0], 'A', 'C', 't4'),
        ([ 0, -1], 'A', 'C', 't4'),
        ([-1,  0], 'A', 'C', 't4'),
        ([-1, -1], 'A', 'C', 't4'),
        ([ 0,  0], 'B', 'D', 't4'),
        ([ 0, -1], 'B', 'D', 't4'),
        ([-1,  0], 'B', 'D', 't4'),
        ([-1, -1], 'B', 'D', 't4'),
        # t5
        ([-1,  0], 'A', 'B', 't5'),
        ([ 0,  1], 'A', 'B', 't5'),
        ([ 0, -1], 'A', 'B', 't5'),
        ([-1,  0], 'C', 'D', 't5'),
        ([ 0,  1], 'C', 'D', 't5'),
        ([ 0, -1], 'C', 'D', 't5'),
    )

    return lat
