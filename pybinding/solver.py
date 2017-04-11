"""Eigensolvers with a few extra computation methods

The :class:`.Solver` class is the main interface for dealing with eigenvalue problems. It
is made to work specifically with pybinding's :class:`.Model` objects, but it may use any
eigensolver algorithm under the hood.

A few different algorithms are provided out of the box: the :func:`.lapack`, :func:`.arpack`
and :func:`.feast` functions return concrete :class:`.Solver` implementation using the LAPACK,
ARPACK and FEAST algorithms, respectively.

The :class:`.Solver` may easily be extended with new eigensolver algorithms. All that is
required is a function which takes a Hamiltonian matrix and returns the computed
eigenvalues and eigenvectors. See :class:`._SolverPythonImpl` for example.
"""
import time
import math

import numpy as np

from . import _cpp
from . import results
from .model import Model
from .system import System

__all__ = ['Solver', 'arpack', 'feast', 'lapack']


class Solver:
    """Computes the eigenvalues and eigenvectors of a Hamiltonian matrix

    This the common interface for various eigensolver implementations. It should not
    be created directly, but via the specific functions: :func:`.lapack`, :func:`.arpack`
    and :func:`.feast`. Those functions will set up their specific solver strategy and
    return a properly configured :class:`.Solver` object.
    """
    def __init__(self, impl: _cpp.Solver):
        self.impl = impl

    @property
    def model(self) -> Model:
        """The tight-binding model attached to this solver"""
        return self.impl.model

    @model.setter
    def model(self, model):
        self.impl.model = model

    @property
    def system(self) -> System:
        """The tight-binding system attached to this solver (shortcut for Solver.model.system)"""
        return System(self.impl.system)

    @property
    def eigenvalues(self) -> np.ndarray:
        """1D array of computed energy states"""
        return self.impl.eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        """2D array where each column represents a wave function

        eigenvectors.shape == (system.num_sites, eigenvalues.size)
        """
        return self.impl.eigenvectors

    def solve(self):
        """Explicitly solve the eigenvalue problem right now

        This method is usually not needed because the main result properties,
        :attr:`.eigenvalues` and :attr:`.eigenvectors`, will call this implicitly
        the first time they are accessed. However, since the :meth:`solve()` routine
        may be computationally expensive, it is useful to have the ability to call it
        ahead of time as needed.
        """
        self.impl.solve()

    def clear(self):
        """Clear the computed results and start over"""
        self.impl.clear()

    def report(self, shortform=False) -> str:
        """Return a report of the last :meth:`solve()` computation

        Parameters
        ----------
        shortform : bool, optional
            Return a short one line version of the report
        """
        return self.impl.report(shortform)

    def set_wave_vector(self, k):
        """Set the wave vector for periodic models

        Parameters
        ----------
        k : array_like
            Wave vector in reciprocal space.
        """
        self.clear()
        self.model.set_wave_vector(k)

    def calc_eigenvalues(self, map_probability_at=None):
        """Return an :class:`.Eigenvalues` result object with an optional probability colormap

        While the :attr:`.eigenvalues` property returns the raw values array, this
        method returns a result object with more data. In addition to the energy
        states, this result may show a colormap of the probability density for each
        state at a single position.

        Parameters
        ----------
        map_probability_at : array_like, optional
            Cartesian position where the probability density of each energy state
            should be calculated.

        Returns
        -------
        :class:`~pybinding.Eigenvalues`
        """
        if not map_probability_at:
            return results.Eigenvalues(self.eigenvalues)
        else:
            site_idx = self.system.find_nearest(position=map_probability_at)
            probability = abs(self.eigenvectors[site_idx, :])**2

            # sum probabilities of degenerate states
            for idx in self.find_degenerate_states(self.eigenvalues):
                probability[idx] = np.sum(probability[idx]) / len(idx)

            return results.Eigenvalues(self.eigenvalues, probability)

    def calc_probability(self, n, reduce=1e-5):
        r"""Calculate the spatial probability density

        .. math::
            \text{P}(r) = |\Psi_n(r)|^2

        for each position :math:`r` in `system.positions` where :math:`\Psi_n(r)`
        is `eigenvectors[:, n]`.

        Parameters
        ----------
        n : int or array_like
            Index of the desired eigenstate. If an array of indices is given, the
            probability will be calculated at each one and a sum will be returned.
        reduce : float, optional
            Reduce degenerate states by summing their probabilities. Neighboring
            states are considered degenerate if their energy is difference is lower
            than the value of `reduce`. This is disabled by passing `reduce=0`.

        Returns
        -------
        :class:`~pybinding.StructureMap`
        """
        if reduce and np.isscalar(n):
            n = np.flatnonzero(abs(self.eigenvalues[n] - self.eigenvalues) < reduce)

        probability = abs(self.eigenvectors[:, n]) ** 2
        if probability.ndim > 1:
            probability = np.sum(probability, axis=1)
        return self.system.with_data(probability)

    def calc_dos(self, energies, broadening):
        r"""Calculate the density of states as a function of energy

        .. math::
            \text{DOS}(E) = \frac{1}{c \sqrt{2\pi}}
                            \sum_n{e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each :math:`E` in `energies`, where :math:`c` is `broadening` and
        :math:`E_n` is `eigenvalues[n]`.

        Parameters
        ----------
        energies : array_like
            Values for which the DOS is calculated.
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS.

        Returns
        -------
        :class:`~pybinding.Series`
        """
        if hasattr(self.impl, 'calc_dos'):
            dos = self.impl.calc_dos(energies, broadening)
        else:
            scale = 1 / (broadening * math.sqrt(2 * math.pi))
            delta = self.eigenvalues[:, np.newaxis] - energies
            dos = scale * np.sum(np.exp(-0.5 * delta**2 / broadening**2), axis=0)
        return results.Series(energies, dos, labels=dict(variable="E (eV)", data="DOS"))

    def calc_ldos(self, energies, broadening, position, sublattice="", reduce=True):
        r"""Calculate the local density of states as a function of energy at the given position

        .. math::
            \text{LDOS}(E) = \frac{1}{c \sqrt{2\pi}}
                             \sum_n{|\Psi_n(r)|^2 e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each :math:`E` in `energies`, where :math:`c` is `broadening`,
        :math:`E_n` is `eigenvalues[n]` and :math:`r` is a single site position
        determined by the arguments `position` and `sublattice`.

        Parameters
        ----------
        energies : array_like
            Values for which the DOS is calculated.
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS.
        position : array_like
            Cartesian position of the lattice site for which the LDOS is calculated.
            Doesn't need to be exact: the method will find the actual site which is
            closest to the given position.
        sublattice : str
            Only look for sites of a specific sublattice, closest to `position`.
            The default value considers any sublattice.
        reduce : bool
            This option is only relevant for multi-orbital models. If true, the
            resulting LDOS will summed over all the orbitals at the target site
            and the result will be a 1D array. If false, the individual orbital
            results will be preserved and the result will be a 2D array with
            `shape == (energy.size, num_orbitals)`.

        Returns
        -------
        :class:`~pybinding.Series`
        """
        if hasattr(self.impl, 'calc_ldos'):
            ldos = self.impl.calc_ldos(energies, broadening, position, sublattice)
        else:
            delta = self.eigenvalues[:, np.newaxis] - energies
            gaussian = np.exp(-0.5 * delta**2 / broadening**2)
            scale = 1 / (broadening * math.sqrt(2 * math.pi))

            sys_idx = self.system.find_nearest(position, sublattice)
            ham_idx = self.system.to_hamiltonian_indices(sys_idx)

            def calc_single(index):
                psi2 = np.abs(self.eigenvectors[index])**2
                return scale * np.sum(psi2[:, np.newaxis] * gaussian, axis=0)

            ldos = np.array([calc_single(i) for i in ham_idx]).T
            if reduce:
                ldos = np.sum(ldos, axis=1)

        return results.Series(energies, ldos.squeeze(), labels=dict(variable="E (eV)", data="LDOS",
                                                                    columns="orbitals"))

    def calc_spatial_ldos(self, energy, broadening):
        r"""Calculate the spatial local density of states at the given energy

        .. math::
            \text{LDOS}(r) = \frac{1}{c \sqrt{2\pi}}
                             \sum_n{|\Psi_n(r)|^2 e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each position :math:`r` in `system.positions`, where :math:`E` is `energy`,
        :math:`c` is `broadening`, :math:`E_n` is `eigenvalues[n]` and :math:`\Psi_n(r)`
        is `eigenvectors[:, n]`.

        Parameters
        ----------
        energy : float
            The energy value for which the spatial LDOS is calculated.
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS.

        Returns
        -------
        :class:`~pybinding.StructureMap`
        """
        if hasattr(self.impl, 'calc_spatial_ldos'):
            ldos = self.impl.calc_spatial_ldos(energy, broadening)
        else:
            scale = 1 / (broadening * math.sqrt(2 * math.pi))
            gaussian = np.exp(-0.5 * (self.eigenvalues - energy)**2 / broadening**2)
            psi2 = np.abs(self.eigenvectors)**2
            ldos = scale * np.sum(psi2 * gaussian, axis=1)

        return self.system.with_data(ldos)

    def calc_bands(self, k0, k1, *ks, step=0.1):
        """Calculate the band structure on a path in reciprocal space

        Parameters
        ----------
        k0, k1, *ks : array_like
            Points in reciprocal space which form the path for the band calculation.
            At least two points are required.
        step : float, optional
            Calculation step length in reciprocal space units. Lower `step` values
            will return more detailed results.

        Returns
        -------
        :class:`~pybinding.Bands`
        """
        k_points = [np.atleast_1d(k) for k in (k0, k1) + ks]
        k_path = results.make_path(*k_points, step=step)

        bands = []
        for k in k_path:
            self.set_wave_vector(k)
            bands.append(self.eigenvalues)

        return results.Bands(k_path, np.vstack(bands))

    @staticmethod
    def find_degenerate_states(energies, abs_tolerance=1e-5):
        """Return groups of indices which belong to degenerate states

        Parameters
        ----------
        energies : array_like
        abs_tolerance : float, optional

        Examples
        --------
        >>> energies = np.array([0.1, 0.1, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.8])
        >>> Solver.find_degenerate_states(energies)
        [[0, 1], [3, 4, 5], [7, 8]]

        >>> energies = np.array([0.1, 0.2, 0.5, 0.7])
        >>> Solver.find_degenerate_states(energies)
        []
        """
        # when:   energy == [0.1, 0.1, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.8]
        # ...     idx == [0, 3, 4, 7]
        idx = np.flatnonzero(abs(np.diff(energies)) < abs_tolerance)
        if idx.size == 0:
            return []
        groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        # ...     groups == [[0], [3, 4], [7]]
        # return: [[0, 1], [3, 4, 5], [7, 8]]
        return [list(g) + [g[-1] + 1] for g in groups]


class _SolverPythonImpl:
    """Python eigensolver implementation

    This is intended to make use of scipy's LAPACK and ARPACK solvers.
    """
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


def lapack(model, **kwargs):
    """LAPACK :class:`.Solver` implementation for dense matrices

    This solver is intended for small models which are best represented by
    dense matrices. Always solves for all the eigenvalues and eigenvectors.
    Internally this solver uses the :func:`scipy.linalg.eigh` function for
    dense Hermitian matrices.

    Parameters
    ----------
    model : Model
        Model which will provide the Hamiltonian matrix.
    **kwargs
        Advanced arguments: forwarded to :func:`scipy.linalg.eigh`.

    Returns
    -------
    :class:`~pybinding.solver.Solver`
    """
    def solver_func(hamiltonian, **kw):
        from scipy.linalg import eigh
        return eigh(hamiltonian.toarray(), **kw)

    return Solver(_SolverPythonImpl(solver_func, model, **kwargs))


def arpack(model, k, sigma=0, **kwargs):
    """ARPACK :class:`.Solver` implementation for sparse matrices

    This solver is intended for large models with sparse Hamiltonian matrices.
    It only computes a small targeted subset of eigenvalues and eigenvectors.
    Internally this solver uses the :func:`scipy.sparse.linalg.eigsh` function
    for sparse Hermitian matrices.

    Parameters
    ----------
    model : Model
        Model which will provide the Hamiltonian matrix.
    k : int
        The desired number of eigenvalues and eigenvectors. This number must be smaller
        than the size of the matrix, preferably much smaller for optimal performance.
        The computed eigenvalues are the ones closest to `sigma`.
    sigma : float, optional
        Look for eigenvalues near `sigma`.
    **kwargs
        Advanced arguments: forwarded to :func:`scipy.sparse.linalg.eigsh`.

    Returns
    -------
    :class:`~pybinding.solver.Solver`
    """
    from scipy.sparse.linalg import eigsh
    if sigma == 0:
        # eigsh can cause problems when sigma is exactly zero
        sigma = np.finfo(model.hamiltonian.dtype).eps
    return Solver(_SolverPythonImpl(eigsh, model, k=k, sigma=sigma, **kwargs))


def feast(model, energy_range, initial_size_guess, recycle_subspace=False, is_verbose=False):
    """FEAST :class:`.Solver` implementation for sparse matrices

    This solver is only available if the C++ extension module was compiled with FEAST.

    Parameters
    ----------
    model : Model
        Model which will provide the Hamiltonian matrix.
    energy_range : tuple of float
        The lowest and highest eigenvalue between which to compute the solutions.
    initial_size_guess : int
        Initial user guess for number of eigenvalues which will be found in the given
        `energy_range`. This value may be completely wrong - the solver will auto-correct
        as needed. However, for optimal performance the estimate should be as close to
        1.5 * actual_size as possible.
    recycle_subspace : bool, optional
        Reuse previously computed values as a starting point for the next computation.
        This improves performance when subsequent computations differ only slightly, as
        is the case for the band structure of periodic systems where the results change
        gradually as a function of the wave vector. It may hurt performance otherwise.
    is_verbose : bool, optional
        Show the raw output from the FEAST routine.

    Returns
    -------
    :class:`~pybinding.solver.Solver`
    """
    try:
        # noinspection PyUnresolvedReferences
        return Solver(_cpp.FEAST(model, energy_range, initial_size_guess,
                                 recycle_subspace, is_verbose))
    except AttributeError:
        raise Exception("The module was compiled without the FEAST solver.\n"
                        "Use a different solver or recompile the module with FEAST.")
