"""Computations based on Chebyshev polynomial expansion

The kernel polynomial method (KPM) can be used to approximate various functions by expanding them
in a series of Chebyshev polynomials.
"""
from . import _cpp
from . import results
from .model import Model
from .system import System

__all__ = ['KernelPolynomialMethod', 'kpm', 'kpm_cuda', 'jackson_kernel', 'lorentz_kernel']


class KernelPolynomialMethod:
    """The common interface for various KPM implementations

    It should not be created directly but via specific functions
    like :func:`kpm` or :func:`kpm_cuda`.
    """

    def __init__(self, impl):
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
        """The tight-binding system (shortcut for `KernelPolynomialMethod.model.system`)"""
        return System(self.impl.system)

    def report(self, shortform=False):
        """Return a report of the last computation

        Parameters
        ----------
        shortform : bool, optional
            Return a short one line version of the report
        """
        return self.impl.report(shortform)

    def __call__(self, *args, **kwargs):
        """Deprecated"""  # TODO: remove
        return self.calc_greens(*args, **kwargs)

    def calc_greens(self, i, j, energy, broadening):
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

    def calc_ldos(self, energy, broadening, position, sublattice=""):
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
        sublattice : str
            Only look for sites of a specific sublattice, closest to `position`.
            The default value considers any sublattice.

        Returns
        -------
        :class:`~pybinding.LDOS`
        """
        ldos = self.impl.calc_ldos(energy, broadening, position, sublattice)
        return results.LDOS(energy, ldos)

    def deferred_ldos(self, energy, broadening, position, sublattice=""):
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
        sublattice : str
            Only look for sites of a specific sublattice, closest to `position`.
            The default value considers any sublattice.

        Returns
        -------
        Deferred
        """
        return self.impl.deferred_ldos(energy, broadening, position, sublattice)


def kpm(model, energy_range=None, kernel="default", **kwargs):
    """The default CPU implementation of the Kernel Polynomial Method

    This implementation works on any system and is well optimized.

    Parameters
    ----------
    model : Model
        Model which will provide the Hamiltonian matrix.
    energy_range : Optional[Tuple[float, float]]
        KPM needs to know the lowest and highest eigenvalue of the Hamiltonian, before
        computing the expansion moments. By default, this is determined automatically
        using a quick Lanczos procedure. To override the automatic boundaries pass a
        `(min_value, max_value)` tuple here. The values can be overestimated, but note
        that performance drops as the energy range becomes wider. On the other hand,
        underestimating the range will produce `NaN` values in the results.
    kernel : Kernel
        The kernel in the *Kernel* Polynomial Method. Used to improve the quality of
        the function reconstructed from the Chebyshev series. Possible values are
        :func:`jackson_kernel` or :func:`lorentz_kernel`. The Lorentz kernel is used
        by default with `lambda = 4`.

    Returns
    -------
    :class:`~pybinding.chebyshev.KernelPolynomialMethod`
    """
    if kernel == "default":
        kernel = lorentz_kernel()
    return KernelPolynomialMethod(_cpp.kpm(model, energy_range or (0, 0), kernel, **kwargs))


def kpm_cuda(model, energy_range=None, kernel="default", **kwargs):
    """Same as :func:`kpm` except that it's executed on the GPU using CUDA (if supported)

    See :func:`kpm` for detailed parameter documentation.
    This method is only available if the C++ extension module was compiled with CUDA.

    Parameters
    ----------
    model : Model
    energy_range : Optional[Tuple[float, float]]
    kernel : Kernel

    Returns
    -------
    :class:`~pybinding.chebyshev.KernelPolynomialMethod`
    """
    try:
        if kernel == "default":
            kernel = lorentz_kernel()
        # noinspection PyUnresolvedReferences
        return KernelPolynomialMethod(_cpp.kpm_cuda(model, energy_range or (0, 0),
                                                    kernel, **kwargs))
    except AttributeError:
        raise Exception("The module was compiled without CUDA support.\n"
                        "Use a different KPM implementation or recompile the module with CUDA.")


def jackson_kernel():
    """The Jackson kernel -- a good general-purpose kernel, appropriate for most applications

    Imposes Gaussian broadening `sigma = pi / N` where `N` is the number of moments. The
    broadening value is user-defined for each function calculation (LDOS, Green's, etc.).
    The number of moments is then determined based on the broadening -- it's not directly
    set by the user.
    """
    return _cpp.jackson_kernel()


def lorentz_kernel(lambda_value=4.0):
    """The Lorentz kernel -- best for Green's function

    This kernel is most appropriate for the expansion of the Green’s function because it most
    closely mimics the divergences near the true eigenvalues of the Hamiltonian. The Lorentzian
    broadening is given by `epsilon = lambda / N` where `N` is the number of moments.

    Parameters
    ----------
    lambda_value : float
        May be used to fine-tune the smoothness of the convergence. Usual values are
        between 3 and 5. Lower values will speed up the calculation at the cost of
        accuracy. If in doubt, leave it at the default value of 4.
    """
    return _cpp.lorentz_kernel(lambda_value)
