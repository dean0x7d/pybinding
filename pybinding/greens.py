"""Green's function computation and related methods

Deprecated: use the chebyshev module instead
"""
import warnings
from . import chebyshev
from .support.deprecated import LoudDeprecationWarning

__all__ = ['Greens', 'kpm', 'kpm_cuda']

Greens = chebyshev.KPM


def kpm(*args, **kwargs):
    warnings.warn("Use pb.kpm() instead", LoudDeprecationWarning, stacklevel=2)
    return chebyshev.kpm(*args, **kwargs)


def kpm_cuda(*args, **kwargs):
    warnings.warn("Use pb.kpm_cuda() instead", LoudDeprecationWarning, stacklevel=2)
    return chebyshev.kpm_cuda(*args, **kwargs)
