import _pybinding
from .plot import utils as pltutils
from .utils import with_defaults
import matplotlib.pyplot as plt
import numpy as np


class LDOSpoint:
    def __init__(self, energy, ldos):
        self.energy = energy
        self.ldos = ldos

    def plot(self, **kwargs):
        plt.plot(self.energy, self.ldos, **kwargs)
        plt.xlim(self.energy.min(), self.energy.max())
        plt.ylabel('LDOS')
        plt.xlabel('E (eV)')
        pltutils.despine()
