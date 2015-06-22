import _pybinding
from .support.sparse import SparseMatrix


class Hamiltonian:
    def __init__(self, impl: _pybinding.Hamiltonian):
        self.impl = impl

    @property
    def matrix(self) -> SparseMatrix:
        return SparseMatrix(self.impl.matrix)
