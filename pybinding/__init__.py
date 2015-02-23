# linux specific code to set proper dlopen flags (MKL doesn't work otherwise)
import sys
if sys.platform.startswith("linux"):
    import os
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from .model import Model
from .lattice import Lattice

from . import lattice, shape, symmetry
from . import system, hamiltonian
from . import solver, greens, result
from . import electric, magnetic
