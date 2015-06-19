from ..solver import Solver

try:
    from _pybinding import FEAST
except ImportError:
    class FEAST:
        def __init__(self, *args, **kwargs):
            raise Exception("The module was compiled without the FEAST solver.\n"
                            "Use a different solver or recompile the module with FEAST.")


def make_feast(model, energy_range, initial_size_guess, recycle_subspace=False, is_verbose=False):
    return Solver(FEAST(model, energy_range, initial_size_guess, recycle_subspace, is_verbose))
