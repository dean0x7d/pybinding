import _pybinding
from .support.sparse import SparseMatrix as _SparseMatrix


class Hamiltonian(_pybinding.Hamiltonian):
    @property
    def matrix(self) -> _SparseMatrix:
        matrix = self._matrix
        matrix.__class__ = _SparseMatrix
        return matrix
