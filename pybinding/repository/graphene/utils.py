

def landau_level(magnetic_field: float, n: int):
    """ Calculate the energy of Landau level n in the given magnetic field. """
    from pybinding.constant import hbar
    from .lattice import vf
    from math import sqrt
    
    lb = sqrt(hbar / magnetic_field)
    return hbar * (vf * 10**-9) / lb * sqrt(2 * n)
