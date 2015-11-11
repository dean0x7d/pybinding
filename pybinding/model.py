import numpy as np
from scipy.sparse import csr_matrix

from . import _cpp
from . import results
from .system import System
from .lattice import Lattice
from .support.sparse import SparseMatrix


class Model(_cpp.Model):
    def __init__(self, *params):
        super().__init__()
        self.add(*params)

    def add(self, *params):
        for param in params:
            if param is None:
                continue

            if isinstance(param, (tuple, list)):
                self.add(*param)
            else:
                super().add(param)

    @property
    def system(self) -> System:
        return System(super().system)

    @property
    def hamiltonian(self) -> csr_matrix:
        matrix = SparseMatrix(super().hamiltonian.matrix)
        return matrix.tocsr()

    @property
    def lattice(self) -> Lattice:
        return super().lattice

    @property
    def modifiers(self) -> list:
        return (self.state_modifiers + self.position_modifiers +
                self.onsite_modifiers + self.hopping_modifiers)

    @property
    def onsite_map(self) -> results.StructureMap:
        """`StructureMap` of the onsite energy"""
        onsite_energy = np.real(self.hamiltonian.tocsr().diagonal())
        return results.StructureMap.from_system(onsite_energy, self.system)
