import _pybinding
from scipy.sparse import csr_matrix as _csrmatrix


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
    def system(self):
        from .system import System as SystemEx
        sys = super().system
        sys.__class__ = SystemEx
        sys.shape = self.shape
        sys.lattice = self.lattice
        return sys

    @property
    def _hamiltonian(self):
        from .hamiltonian import Hamiltonian as HamiltonianEx
        ham = super().hamiltonian
        ham.__class__ = HamiltonianEx
        return ham

    @property
    def hamiltonian(self) -> _csrmatrix:
        from .hamiltonian import Hamiltonian as HamiltonianEx
        ham = super().hamiltonian
        ham.__class__ = HamiltonianEx
        return ham.matrix.to_scipy_csr()

    @property
    def solver(self):
        from .solver.solver_ex import SolverEx
        sol = super().solver
        sol.__class__ = SolverEx
        sol.system = self.system
        return sol
