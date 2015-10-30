import time
import math

import numpy as np

from . import _cpp
from . import results
from .system import System
from .support.pickle import pickleable

__all__ = ['Solver', 'make_feast', 'make_lapack', 'make_arpack']


@pickleable(impl='system. eigenvalues eigenvectors')
class Solver:
    def __init__(self, impl: _cpp.Solver):
        self.impl = impl

    def solve(self):
        self.impl.solve()

    def clear(self):
        self.impl.clear()

    def report(self, shortform=False):
        return self.impl.report(shortform)

    def set_wave_vector(self, k):
        self.clear()
        self.model.set_wave_vector(k)

    @property
    def model(self):
        return self.impl.model

    @model.setter
    def model(self, model):
        self.impl.model = model

    @property
    def system(self) -> System:
        return System(self.impl.system)

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.impl.eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.impl.eigenvectors

    def calc_eigenvalues(self, map_probability_at=None):
        if not map_probability_at:
            return results.Eigenvalues(self.eigenvalues)
        else:
            site_idx = self.system.find_nearest(position=map_probability_at)
            probability = abs(self.eigenvectors[site_idx, :])**2

            # sum probabilities of degenerate states
            for idx in self.find_degenerate_states(self.eigenvalues):
                probability[idx] = np.sum(probability[idx]) / len(idx)

            return results.Eigenvalues(self.eigenvalues, probability)

    def calc_probability(self, indices, reduce=1e-5):
        if reduce and np.isscalar(indices):
            indices = np.flatnonzero(abs(self.eigenvalues[indices] - self.eigenvalues) < reduce)

        probability = abs(self.eigenvectors[:, indices]) ** 2
        if probability.ndim > 1:
            probability = np.sum(probability, axis=1)
        return results.StructureMap.from_system(probability, self.system)

    def calc_dos(self, energies, broadening) -> np.ndarray:
        """Calculate the density of states as a function of energy

        Calculates the following, where E is `energies` and c is `broadening`
            DOS(E) = 1 / (c * sqrt(2pi)) * sum(exp(-0.5 * (eigenvalues - E)^2 / c^2))

        Parameters
        ----------
        energies : array_like
            Values for which the DOS is calculated.
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS.

        Returns
        -------
        results.DOS
        """
        if hasattr(self.impl, 'calc_dos'):
            return results.DOS(energies, self.impl.calc_dos(energies, broadening))
        else:
            scale = 1 / (broadening * math.sqrt(2 * math.pi))
            delta = self.eigenvalues[:, np.newaxis] - energies
            dos = scale * np.sum(np.exp(-0.5 * delta**2 / broadening**2), axis=0)
            return results.DOS(energies, dos)

    def calc_ldos(self, energy, broadening, sublattice=-1):
        ldos = self.impl.calc_ldos(energy, broadening, sublattice)
        return results.SpatialMap.from_system(ldos, self.system)

    def calc_bands(self, k0, k1, *ks, step=0.1):
        k_points = [np.atleast_1d(k) for k in (k0, k1) + ks]
        k_path = results.make_path(*k_points, step=step)

        bands = []
        for k in k_path:
            self.set_wave_vector(k)
            bands.append(self.eigenvalues)

        return results.Bands(k_points, k_path, np.vstack(bands))

    @staticmethod
    def find_degenerate_states(energies, abs_tolerance=1e-5):
        """Return groups of indices which belong to degenerate states

        >>> energies = np.array([0.1, 0.1, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.8])
        >>> Solver.find_degenerate_states(energies)
        [[0, 1], [3, 4, 5], [7, 8]]
        """
        # see doctest for example arg and return
        idx = np.flatnonzero(abs(np.diff(energies)) < abs_tolerance)
        # idx = [1, 0, 0, 1, 1, 0, 0, 1]
        groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        # groups = [[0], [3, 4], [7]]
        return [list(g) + [g[-1] + 1] for g in groups]


class SolverPythonImpl:
    def __init__(self, solve_func, model, **kwargs):
        self.solve_func = solve_func
        self._model = model

        self.kwargs = kwargs
        self.vals = np.empty(0)
        self.vecs = np.empty(0)
        self.compute_time = .0

    def clear(self):
        self.vals = np.empty(0)
        self.vecs = np.empty(0)
        self.compute_time = .0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self.clear()
        self._model = model

    @property
    def system(self):
        return self.model.system.impl

    @property
    def eigenvalues(self) -> np.ndarray:
        self.solve()
        return self.vals

    @property
    def eigenvectors(self) -> np.ndarray:
        self.solve()
        return self.vecs

    def solve(self):
        if len(self.vals):
            return

        start_time = time.time()

        self.vals, self.vecs = self.solve_func(self.model.hamiltonian, **self.kwargs)
        idx = self.vals.argsort()
        self.vals = self.vals[idx]
        self.vecs = self.vecs[:, idx]

        self.compute_time = time.time() - start_time

    def report(self, _=False):
        from .utils.time import pretty_duration
        return "Converged in " + pretty_duration(self.compute_time)


def make_lapack(model, **kwargs):
    from scipy.linalg import eigh
    solver_func = lambda m, **kw: eigh(m.toarray(), **kw)
    return Solver(SolverPythonImpl(solver_func, model, **kwargs))


def make_arpack(model, num_eigenvalues, sigma=1e-5, **kwargs):
    from scipy.sparse.linalg import eigsh
    return Solver(SolverPythonImpl(eigsh, model, k=num_eigenvalues, sigma=sigma, **kwargs))


def make_feast(model, energy_range, initial_size_guess, recycle_subspace=False, is_verbose=False):
    try:
        return Solver(_cpp.FEAST(model, energy_range, initial_size_guess,
                                       recycle_subspace, is_verbose))
    except AttributeError:
        raise Exception("The module was compiled without the FEAST solver.\n"
                        "Use a different solver or recompile the module with FEAST.")
