import _pybinding
import numpy as _np


class OnsiteModifier(_pybinding.OnsiteModifier):
    def is_complex(self):
        one, zero = _np.ones(1), _np.zeros(1)
        v = self.apply(one, zero, zero, zero)
        return v.dtype == _np.complex_


class HoppingModifier(_pybinding.HoppingModifier):
    def is_complex(self):
        one, zero = _np.ones(1), _np.zeros(1)
        t = self.apply(one, zero, zero, zero, zero, zero, zero)
        return t.dtype == _np.complex_


def _make_modifier_decorator(base):
    def decorator(f):
        def wrapper():
            class Derived(base):
                # noinspection PyMethodMayBeStatic
                def apply(self, *args):
                    return f(*args)
            return Derived()
        return wrapper()
    return decorator


site_state = _make_modifier_decorator(_pybinding.SiteStateModifier)
site_position = _make_modifier_decorator(_pybinding.PositionModifier)
onsite_energy = _make_modifier_decorator(OnsiteModifier)
hopping_energy = _make_modifier_decorator(HoppingModifier)
