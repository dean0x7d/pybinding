from .__about__ import (__author__, __copyright__, __doc__, __email__, __license__, __summary__,
                        __title__, __url__, __version__)

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

import _pybinding as _cpp

from .model import Model
from .lattice import *
from .shape import *
from .modifier import *

from .results import make_path
from .support.pickle import save, load
from .parallel import parallel_for, parallelize

from . import (constants, greens, parallel, pltutils, results, solver, system, utils)
