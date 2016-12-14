"""Green's function computation and related methods

Deprecated: use the chebyshev module instead
"""
from .chebyshev import KernelPolynomialMethod, kpm, kpm_cuda

__all__ = ['Greens', 'kpm', 'kpm_cuda']

Greens = KernelPolynomialMethod
