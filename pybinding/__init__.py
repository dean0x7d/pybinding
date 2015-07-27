import os
import sys
if sys.platform.startswith("linux"):
    # In case the pybinding C++ extension is compiled with MKL, it will not play nice
    # with sparse.linalg (segfaults). As a workaround, sparse.linalg is imported first
    # with default dlopenflags.
    # After that RTLD_GLOBAL must be set for MKL to load properly. It's not possible
    # to set RTLD_GLOBAL, import _pybinding (with MKL) and then reset to default flags.
    import scipy.sparse.linalg
    sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)

from .model import Model
from .lattice import Lattice, make_lattice
from .results import make_path

from . import (constants, electric, greens, lattice, magnetic, model, modifier,
               parallel, pltutils, results, shape, solver, symmetry, system)
