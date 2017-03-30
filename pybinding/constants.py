"""A few useful physical constants

Note that energy is expressed in units of eV.
"""
from math import pi
import numpy as np

c = 299792458  #: [m/s] speed of light
e = 1.602 * 10**-19  #: [C] electron charge
epsilon0 = 8.854 * 10**-12  #: [F/m] vacuum permittivity
hbar = 6.58211899 * 10**-16  #: [eV*s] reduced Plank constant
phi0 = 2 * pi * hbar  #: [V*s] magnetic quantum


class Pauli:
    x = np.array([[0, 1],
                  [1, 0]])
    y = np.array([[0, -1j],
                  [1j,  0]])
    z = np.array([[1,  0],
                  [0, -1]])

    def __repr__(self):
        return "x: [[0, 1], [1, 0]], y: [[0, -1j], [1j, 0]], z: [[1, 0], [0, -1]]"


pauli = Pauli()  #: Pauli matrices -- use the ``.x``, ``.y`` and ``.z`` attributes
