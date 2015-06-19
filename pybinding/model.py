import _pybinding
from scipy.sparse import csr_matrix
from .system import System
from .hamiltonian import Hamiltonian
from .solver import Solver


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
    def system(self) -> System:
        sys = super().system
        sys.__class__ = System
        return sys

    @property
    def _hamiltonian(self) -> Hamiltonian:
        ham = super().hamiltonian
        ham.__class__ = Hamiltonian
        return ham

    @property
    def hamiltonian(self) -> csr_matrix:
        ham = super().hamiltonian
        ham.__class__ = Hamiltonian
        return ham.matrix.tocsr()

    def plot_bands(self, k0, k1, *ks, step, names=None):
        # TODO: move into Solver
        import numpy as np
        import matplotlib.pyplot as plt

        ks = [np.array(k) for k in (k0, k1) + ks]
        energy = []
        points = [0]
        for start, end in zip(ks[:-1], ks[1:]):
            num_steps = max(abs(end - start) / step)
            k_list = (np.linspace(s, e, num_steps) for s, e in zip(start, end))

            for k in zip(*k_list):
                self.set_wave_vector(k)
                energy.append(self.solver.eigenvalues.copy())
            points += [len(energy)-1]

        for point in points[1:-1]:
            plt.axvline(point, color='black', ls='--')
        plt.xticks(points, names if names else [])

        plt.plot(energy, color='blue')
        plt.xlim(0, len(energy) - 1)
        plt.xlabel('k-space')
        plt.ylabel('E (eV)')
