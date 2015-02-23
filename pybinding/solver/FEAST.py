try:
    from _pybinding import FEAST
except ImportError:
    class FEAST:
        def __init__(self, *args, **kwargs):
            raise Exception("The module was compiled without the FEAST solver.\n"
                            "Use a different solver or recompile the module with FEAST.")
