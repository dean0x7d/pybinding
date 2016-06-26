"""Green's function computation and related methods

The main approach uses the Kernel Polynomial Method. This is the only approach
which is implemented at this time, but the :class:`Greens` interface is quite
easy to extend with new algorithms.
"""
from . import _cpp
from . import results
from .model import Model
from .system import System

__all__ = ['Greens', 'kpm', 'kpm_cuda']


class Greens:
    """Computes the Green's function of a Hamiltonian matrix

    This the common interface for various implementations. It should not be
    created directly but via specific functions like :func:`kpm`.
    """
    def __init__(self, impl: _cpp.Greens):
        self.impl = impl

    @property
    def model(self) -> Model:
        """The tight-binding model holding the Hamiltonian"""
        return self.impl.model

    @model.setter
    def model(self, model):
        self.impl.model = model

    @property
    def system(self) -> System:
        """The tight-binding system (shortcut for Greens.model.system)"""
        return System(self.impl.system)

    def report(self, shortform=False):
        """Return a report of the last computation

        Parameters
        ----------
        shortform : bool, optional
            Return a short one line version of the report
        """
        return self.impl.report(shortform)

    def __call__(self, i, j, energy, broadening):
        """Calculate Green's function of a single Hamiltonian element

        Parameters
        ----------
        i, j : int
            Hamiltonian indices.
        energy : ndarray
            Energy value array.
        broadening : float
            Width, in energy, of the smallest detail which can be resolved.
            Lower values result in longer calculation time.

        Returns
        -------
        ndarray
            Array of the same size as the input `energy`.
        """
        return self.impl.calc_greens(i, j, energy, broadening)

    def calc_ldos(self, energy, broadening, position, sublattice=None):
        """Calculate the local density of states as a function of energy

        Parameters
        ----------
        energy : ndarray
            Values for which the LDOS is calculated.
        broadening : float
            Width, in energy, of the smallest detail which can be resolved.
            Lower values result in longer calculation time.
        position : array_like
            Cartesian position of the lattice site for which the LDOS is calculated.
            Doesn't need to be exact: the method will find the actual site which is
            closest to the given position.
        sublattice : Optional[int]
            Only look for sites of a specific sublattice, closest to `position`.
            The default value considers any sublattice.

        Returns
        -------
        :class:`~pybinding.LDOS`
        """
        sublattice = self.system.lattice[sublattice] if sublattice is not None else -1
        ldos = self.impl.calc_ldos(energy, broadening, position, sublattice)
        return results.LDOS(energy, ldos)

    def deferred_ldos(self, energy, broadening, position, sublattice=None):
        """Same as :meth:`calc_ldos` but for parallel computation: see the :mod:`.parallel` module

        Parameters
        ----------
        energy : ndarray
            Values for which the LDOS is calculated.
        broadening : float
            Width, in energy, of the smallest detail which can be resolved.
            Lower values result in longer calculation time.
        position : array_like
            Cartesian position of the lattice site for which the LDOS is calculated.
            Doesn't need to be exact: the method will find the actual site which is
            closest to the given position.
        sublattice : Optional[int]
            Only look for sites of a specific sublattice, closest to `position`.
            The default value considers any sublattice.

        Returns
        -------
        Deferred
        """
        sublattice = self.system.lattice[sublattice] if sublattice is not None else -1
        deferred = self.impl.deferred_ldos(energy, broadening, position, sublattice)
        deferred.model = self.model
        return deferred


def kpm(model, lambda_value=4.0, energy_range=None, optimization_level=3, lanczos_precision=0.002):
    """Calculate Green's function using the Kernel Polynomial Method

    Parameters
    ----------
    model : Model
        Model which will provide the Hamiltonian matrix.
    lambda_value : float
        Controls the accuracy of the kernel polynomial method. Usual values are
        between 3 and 5. Lower values will speed up the calculation at the cost
        of accuracy. If in doubt, leave it at the default value of 4.
    energy_range : tuple of float, optional
        KPM needs to know the lowest and highest eigenvalue of the Hamiltonian,
        before computing Green's. By default, this is determined automatically
        using a quick Lanczos procedure. To override the automatic boundaries pass
        a (min_value, max_value) tuple here. The values can be overestimated, but
        it will result in lower performance. However, underestimating the values
        will return NaN results.
    optimization_level : int
        Level 0 disables all optimizations. Level 1 turns on matrix reordering which
        allows some parts of the sparse matrix-vector multiplication to be discarded.
        Level 2 enables moment interleaving: two KPM moments will be calculated per
        iteration which significantly lowers the required memory bandwidth. Level 3
        converts the Hamiltonian matrix format from CSR to ELLPACK format which
        allows for better vectorization of sparse matrix-vector multiplication.
    lanczos_precision : float
        How precise should the automatic Hamiltonian bounds determination be.
        TODO: implementation detail. Remove from public interface.

    Returns
    -------
    :class:`~pybinding.greens.Greens`
    """
    kpm_implementation = _cpp.KPM(model, lambda_value, energy_range or (0, 0),
                                  optimization_level, lanczos_precision)
    return Greens(kpm_implementation)


def kpm_cuda(model, lambda_value=4.0, energy_range=None, optimization_level=1):
    """Same as :func:`kpm` except that it's executed on the GPU using CUDA (if supported)

    See :func:`kpm` for detailed parameter documentation.
    This method is only available if the C++ extension module was compiled with CUDA.

    Parameters
    ----------
    model : Model
    lambda_value : float
    energy_range : Tuple[float]
    optimization_level : int

    Returns
    -------
    :class:`~pybinding.greens.Greens`
    """
    try:
        # noinspection PyUnresolvedReferences
        kpm_implementation = _cpp.KPMcuda(model, lambda_value, energy_range or (0, 0),
                                          optimization_level)
        return Greens(kpm_implementation)
    except AttributeError:
        raise Exception("The module was compiled without CUDA support.\n"
                        "Use a different KPM implementation or recompile the module with CUDA.")

