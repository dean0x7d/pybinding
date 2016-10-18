from .__about__ import (__author__, __copyright__, __doc__, __email__, __license__, __summary__,
                        __title__, __url__, __version__)

import os
import sys
if sys.platform.startswith("linux"):
    # When the _pybinding C++ extension is compiled with MKL, it requires specific
    # dlopen flags on Linux: RTLD_GLOBAL. This will not play nice with some scipy
    # modules, i.e. it will produce segfaults. As a workaround, specific modules
    # are imported first with default dlopenflags.
    # After that, RTLD_GLOBAL must be set for MKL to load properly. It's not possible
    # to set RTLD_GLOBAL, import _pybinding and then reset to default flags. This is
    # fundamentally an MKL issue which makes it difficult to resolve. This workaround
    # is the best solution at the moment.
    import scipy.sparse.linalg
    import scipy.spatial
    sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)

import _pybinding as _cpp

from .model import *
from .lattice import *
from .shape import *
from .modifier import *
from .results import *

from .support.pickle import save, load
from .parallel import parallel_for, parallelize

from . import (constants, greens, parallel, pltutils, results, solver, system, utils)


def tests(options=None, plugins=None):
    """Run the tests

    Parameters
    ----------
    options : list or str
        Command line options for pytest (excluding target file_or_dir).
    plugins : list
        Plugin objects to be auto-registered during initialization.
    """
    import pytest
    import pathlib
    import matplotlib as mpl
    from .utils.misc import cd

    args = options or []
    if isinstance(args, str):
        args = args.split()
    module_path = pathlib.Path(__file__).parent

    if (module_path / 'tests').exists():
        # tests are inside installed package -> use read-only mode
        args.append('--failpath=' + os.getcwd() + '/failed')
        with cd(module_path), pltutils.backend('Agg'):
            args += ['-c', str(module_path / 'tests/local.cfg'), str(module_path)]
            return pytest.main(args, plugins)
    else:
        # tests are in dev environment -> use development mode
        with cd(module_path.parent), pltutils.backend('Agg'):
            return pytest.main(args, plugins)
