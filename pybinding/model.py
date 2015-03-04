import _pybinding
from scipy.sparse import csr_matrix as _csrmatrix
from .system import System as _System
from .hamiltonian import Hamiltonian as _Hamiltonian
from .solver.solver_ex import SolverEx as _Solver


class Model(_pybinding.Model):
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

    def calculate(self, result):
        self._calculate(result)
        return result

    @property
    def system(self) -> _System:
        sys = super().system
        sys.__class__ = _System
        sys.shape = self.shape
        sys.lattice = self.lattice
        return sys

    @property
    def _hamiltonian(self) -> _Hamiltonian:
        ham = super().hamiltonian
        ham.__class__ = _Hamiltonian
        return ham

    @property
    def hamiltonian(self) -> _csrmatrix:
        ham = super().hamiltonian
        ham.__class__ = _Hamiltonian
        return ham.matrix.to_scipy_csr()

    @property
    def solver(self) -> _Solver:
        sol = super().solver
        sol.__class__ = _Solver
        sol.system = self.system
        return sol
