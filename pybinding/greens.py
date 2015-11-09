import numpy as np

from . import _cpp
from .results import LDOSpoint
from .model import Model
from .system import System

__all__ = ['Greens', 'kpm']


class Greens:
    def __init__(self, impl: _cpp.Greens):
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

    def calc_ldos(self, energy: np.ndarray, broadening: float, position: list, sublattice=-1):
        return LDOSpoint(energy, self.impl.calc_ldos(energy, broadening, position, sublattice))

    def deferred_ldos(self, energy: np.ndarray, broadening: float, position: list, sublattice=-1):
        deferred = self.impl.deferred_ldos(energy, broadening, position, sublattice)
        deferred.model = self.model
        return deferred


def kpm(model, lambda_value=4.0, energy_range=(0.0, 0.0),
        optimization_level=2, lanczos_precision=0.002):
    return Greens(_cpp.KPM(model, lambda_value, energy_range,
                           optimization_level, lanczos_precision))
