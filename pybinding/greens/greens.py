import _pybinding
from ..result import LDOSpoint

import numpy as np


class Greens:
    def __init__(self, impl: _pybinding.Greens):
        self.impl = impl

    def calc_ldos(self, energy: np.ndarray, broadening: float, position: tuple, sublattice: int=-1):
        return LDOSpoint(energy, self.impl.calc_ldos(energy, broadening, position, sublattice))
