"""Main model definition interface"""
import numpy as np
from scipy.sparse import csr_matrix

from . import _cpp
from . import results
from .system import System
from .lattice import Lattice
from .support.sparse import SparseMatrix


class Model(_cpp.Model):
    """Builds a tight-binding Hamiltonian from a model description

    The most important properties are :attr:`.system` and :attr:`.hamiltonian` which are
    constructed based on the input parameters. The :class:`.System` contains structural
    data like site positions. The tight-binding Hamiltonian is a sparse matrix in the
    :class:`.scipy.csr_matrix` format.

    The main class implementation is in C++ via the `_cpp.Model` base class.

    Parameters
    ----------
    lattice : Lattice
        The lattice specification.
    *args
        Can be any of: shape, symmetry or various modifiers. Note that:

        * There can be at most one shape and at most one symmetry. Shape and symmetry
          can be composed as desired, but physically impossible scenarios will result
          in an empty system and Hamiltonian.
        * Any number of modifiers can be added. Adding the same modifier more than once
          is allowed: this will usually multiply the modifier's effect.
    """
    def __init__(self, lattice, *args):
        super().__init__(lattice)

        self._lattice = lattice
        self._shape = None
        self.add(*args)

    def add(self, *args):
        """Add parameter(s) to the model

        Parameters
        ----------
        *args
            Any of: shape, symmetry, modifiers. Tuples and lists of parameters are expanded
            automatically, so `M.add(p0, [p1, p2])` is equivalent to `M.add(p0, p1, p2)`.
        """
        for arg in filter(None, args):
            if isinstance(arg, (tuple, list)):
                self.add(*arg)
            else:
                super().add(arg)
                if isinstance(arg, _cpp.Shape):
                    self._shape = arg

    def attach_lead(self, direction, *where):
        """Attach a lead to the main system region

        Not valid for 1D lattices.

        Parameters
        ----------
        direction : int
            Lattice vector direction of the lead. Must be one of: 1, 2, 3, -1, -2, -3.
            For example, `direction=2` would create a lead which intersects the main system
            in the :math:`a_2` lattice vector direction. Setting `direction=-2` would create
            a lead on the opposite side of the system, but along the same lattice vector.
        *where
            Where the lead should be placed:

            * For 2D lattices: Two points between which the lead should pass.
            * For 3D lattices: A :class:`.FreeformShape` defining the 2D area of the lead.
        """
        if len(where) == 1 and isinstance(where[0], _cpp.Shape):
            super().attach_lead(direction, where[0])
        elif len(where) == 2:
            super().attach_lead(direction, _cpp.Line(*where))
        else:
            raise RuntimeError("Bad arguments")

    @property
    def system(self) -> System:
        """:class:`.System` site positions and other structural data"""
        return System(super().system)

    @property
    def hamiltonian(self) -> csr_matrix:
        """Hamiltonian sparse matrix in the :class:`.scipy.csr_matrix` format"""
        matrix = SparseMatrix(super().hamiltonian.matrix)
        return matrix.tocsr()

    @property
    def lattice(self) -> Lattice:
        """:class:`.Lattice` specification"""
        return self._lattice

    @property
    def shape(self):
        """:class:`.Polygon` or :class:`.FreeformShape`"""
        return self._shape

    @property
    def modifiers(self) -> list:
        """List of all modifiers applied to this model"""
        return (self.state_modifiers + self.position_modifiers +
                self.onsite_modifiers + self.hopping_modifiers)

    @property
    def onsite_map(self) -> results.StructureMap:
        """:class:`.StructureMap` of the onsite energy"""
        onsite_energy = np.real(self.hamiltonian.tocsr().diagonal())
        return results.StructureMap.from_system(onsite_energy, self.system)
