"""Main model definition interface"""
import numpy as np
from scipy.sparse import csr_matrix

from . import _cpp
from . import results
from .system import System
from .lattice import Lattice
from .support.sparse import SparseMatrix


class Model(_cpp.Model):
    """Takes tight-binding model parameters and creates a Hamiltonian matrix

    The class is initialized with parameters which can be any of: lattice, shape,
    symmetry or various modifiers. Note that:

    * A `Model` must have one and only one lattice. If multiple are added, only
      the last one is considered.
    * There can be at most one shape and at most one symmetry. Shape and symmetry
      can be composed as desired, but physically impossible scenarios will result
      in an empty system and Hamiltonian.
    * Any number of modifiers can be added. Duplicates are also allowed: the usual
      result being a doubling of the modifier's effect.

    The main properties are `system` and `hamiltonian` which are constructed based
    on the parameters. The Hamiltonian is a sparse matrix in the `scipy.csr_matrix`
    format. The `System` contains structural data of the model. See the `System`
    class for more details.

    The main class implementation is in C++ via the `_cpp.Model` base class.
    """
    def __init__(self, *params):
        super().__init__()
        self.add(*params)

    def add(self, *params):
        """Add parameter(s) to the model

        Parameters
        ----------
        *params
            Any of: lattice, shape, symmetry, modifiers. Tuples and lists of
            parameters are expanded automatically, so `M.add(p0, [p1, p2])`
            is equivalent to `M.add(p0, p1, p2)`.
        """
        for param in filter(None, params):
            if isinstance(param, (tuple, list)):
                self.add(*param)
            else:
                super().add(param)

    @property
    def system(self) -> System:
        """Tight-binding system structure"""
        return System(super().system)

    @property
    def hamiltonian(self) -> csr_matrix:
        """Hamiltonian sparse matrix"""
        matrix = SparseMatrix(super().hamiltonian.matrix)
        return matrix.tocsr()

    @property
    def lattice(self) -> Lattice:
        """Lattice specification"""
        return super().lattice

    @property
    def modifiers(self) -> list:
        """List of all modifiers applied to this model"""
        return (self.state_modifiers + self.position_modifiers +
                self.onsite_modifiers + self.hopping_modifiers)

    @property
    def onsite_map(self) -> results.StructureMap:
        """`StructureMap` of the onsite energy"""
        onsite_energy = np.real(self.hamiltonian.tocsr().diagonal())
        return results.StructureMap.from_system(onsite_energy, self.system)
