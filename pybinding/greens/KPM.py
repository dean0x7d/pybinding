import _pybinding


class KPM(_pybinding.KPM):
    def advanced(self, use_reordering=True, lanczos_precision=0.002, scaling_tolerance=0.01):
        super().advanced(use_reordering, lanczos_precision, scaling_tolerance)
        return self
