"""A single layer of black phosphorus"""
from math import pi, sin, cos
import pybinding as pb


def monolayer_4band(num_hoppings=5):
    """Monolayer phosphorene lattice using the four-band model

    Parameters
    ----------
    num_hoppings : int
        Number of hopping terms to consider: from t2 to t5.
    """
    a = 0.222  # nm
    ax = 0.438  # nm
    ay = 0.332  # nm
    theta = 96.79 * (pi / 180)
    phi = 103.69 * (pi / 180)

    lat = pb.Lattice(a1=[ax, 0], a2=[0, ay])

    h = a * sin(phi - pi / 2)
    s = 0.5 * ax - a * cos(theta / 2)
    lat.add_sublattices(('A', [-s/2,        -ay/2, h], 0),
                        ('B', [ s/2,        -ay/2, 0], 0),
                        ('C', [-s/2 + ax/2,     0, 0], 0),
                        ('D', [ s/2 + ax/2,     0, h], 0))

    lat.register_hopping_energies({'t1': -1.22, 't2': 3.665, 't3': -0.205,
                                   't4': -0.105, 't5': -0.055})

    if num_hoppings < 2:
        raise RuntimeError("t1 and t2 must be included")
    elif num_hoppings > 5:
        raise RuntimeError("t5 is the last one")

    if num_hoppings >= 2:
        lat.add_hoppings(([-1,  0], 'A', 'D', 't1'),
                         ([-1, -1], 'A', 'D', 't1'),
                         ([ 0,  0], 'B', 'C', 't1'),
                         ([ 0, -1], 'B', 'C', 't1'))
        lat.add_hoppings(([ 0,  0], 'A', 'B', 't2'),
                         ([ 0,  0], 'C', 'D', 't2'))
    if num_hoppings >= 3:
        lat.add_hoppings(([ 0,  0], 'A', 'D', 't3'),
                         ([ 0, -1], 'A', 'D', 't3'),
                         ([ 1,  1], 'C', 'B', 't3'),
                         ([ 1,  0], 'C', 'B', 't3'))
    if num_hoppings >= 4:
        lat.add_hoppings(([ 0,  0], 'A', 'C', 't4'),
                         ([ 0, -1], 'A', 'C', 't4'),
                         ([-1,  0], 'A', 'C', 't4'),
                         ([-1, -1], 'A', 'C', 't4'),
                         ([ 0,  0], 'B', 'D', 't4'),
                         ([ 0, -1], 'B', 'D', 't4'),
                         ([-1,  0], 'B', 'D', 't4'),
                         ([-1, -1], 'B', 'D', 't4'))
    if num_hoppings >= 5:
        lat.add_hoppings(([-1,  0], 'A', 'B', 't5'),
                         ([-1,  0], 'C', 'D', 't5'))

    lat.min_neighbors = 2
    return lat
