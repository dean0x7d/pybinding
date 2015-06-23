from _pybinding import KPM
from .greens import Greens


def make_kpm(model, lambda_value=4.0, energy_range=(0.0, 0.0)):
    return Greens(KPM(model, lambda_value, energy_range))
