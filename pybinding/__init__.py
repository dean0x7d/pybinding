# linux specific code to set proper dlopen flags (MKL doesn't work otherwise)
import sys
if sys.platform.startswith("linux"):
    import os
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from .model import Model
from .lattice import Lattice, make_lattice
from .results import make_path

from . import (constants, electric, greens, lattice, magnetic, model,
               modifier, results, shape, solver, symmetry, system, pltutils)
