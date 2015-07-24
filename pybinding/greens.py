import numpy as np

import _pybinding
from .results import LDOSpoint
from .model import Model
from .system import System

__all__ = ['Greens', 'make_kpm']


class Greens:
    def __init__(self, impl: _pybinding.Greens):
        self.impl = impl

    @property
    def model(self) -> Model:
        return self.impl.model

    @model.setter
    def model(self, model):
        self.impl.model = model

    @property
    def system(self) -> System:
        return System(self.impl.system)

    def report(self, shortform=False):
        return self.impl.report(shortform)

    def calc_ldos(self, energy: np.ndarray, broadening: float, position: tuple, sublattice: int=-1):
        return LDOSpoint(energy, self.impl.calc_ldos(energy, broadening, position, sublattice))


def make_kpm(model, lambda_value=4.0, energy_range=(0.0, 0.0)):
    return Greens(_pybinding.KPM(model, lambda_value, energy_range))
