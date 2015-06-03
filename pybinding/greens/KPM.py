import _pybinding
from ..result import LDOSpoint

import numpy as np


class KPM(_pybinding.KPM):
    def advanced(self, use_reordering=True, lanczos_precision=0.002, scaling_tolerance=0.01):
        super().advanced(use_reordering, lanczos_precision, scaling_tolerance)
        return self

    def calc_ldos(self, energy: np.ndarray, broadening: float, position: tuple, sublattice: int=-1):
        return LDOSpoint(energy, super().calc_ldos(energy, broadening, position, sublattice))
