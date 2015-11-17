"""Example components: lattices, shapes and modifiers

Components which aren't very useful for simulations,
but great for examples and testing.
"""
import pybinding as pb


def square_lattice(d=0.2, t=-1):
    """A simple square lattice

    Parameters
    ----------
    d : float
        Length of the unit cell.
    t : float
        Hopping energy.

    Returns
    -------
    pb.Lattice
    """
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_one_sublattice('A', [0, 0])
    lat.add_hoppings(
        ([0,  1], 'A', 'A', t),
        ([1,  0], 'A', 'A', t),
        ([1,  1], 'A', 'A', t),
        ([1, -1], 'A', 'A', t),
    )
    return lat
