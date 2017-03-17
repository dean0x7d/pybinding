"""Main model definition interface"""
import numpy as np
from scipy.sparse import csr_matrix

from . import _cpp
from . import results
from .system import System, decorate_structure_plot
from .lattice import Lattice
from .leads import Leads

__all__ = ['Model']


class Model(_cpp.Model):
    """Builds a Hamiltonian from lattice, shape, symmetry and modifier parameters

    The most important attributes are :attr:`.system` and :attr:`.hamiltonian` which are
    constructed based on the input parameters. The :class:`.System` contains structural
    data like site positions. The tight-binding Hamiltonian is a sparse matrix in the
    :class:`.scipy.sparse.csr_matrix` format.

    Parameters
    ----------
    lattice : :class:`~pybinding.Lattice`
        The lattice specification.
    *args
        Can be any of: shape, symmetry or various modifiers. Note that:

        * There can be at most one shape and at most one symmetry. Shape and symmetry
          can be composed as desired, but physically impossible scenarios will result
          in an empty system.
        * Any number of modifiers can be added. Adding the same modifier more than once
          is allowed: this will usually multiply the modifier's effect.
    """
    def __init__(self, lattice, *args):
        super().__init__(lattice.impl)

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
        for arg in args:
            if arg is None:
                raise RuntimeError("`None` was passed to Model: check that all "
                                   "modifier functions have return values")
            try:
                self.add(*arg)
            except TypeError:
                super().add(arg)
                if isinstance(arg, _cpp.Shape):
                    self._shape = arg

    def attach_lead(self, direction, contact):
        """Attach a lead to the main system

        Not valid for 1D lattices.

        Parameters
        ----------
        direction : int
            Lattice vector direction of the lead. Must be one of: 1, 2, 3, -1, -2, -3.
            For example, `direction=2` would create a lead which intersects the main system
            in the :math:`a_2` lattice vector direction. Setting `direction=-2` would create
            a lead on the opposite side of the system, but along the same lattice vector.
        contact : :class:`~_pybinding.Shape`
            The place where the lead should contact the main system. For a 2D lattice it's
            just a :func:`.line` describing the intersection of the lead and the system.
            For a 3D lattice it's the area described by a 2D :class:`.FreeformShape`.
        """
        super().attach_lead(direction, contact)

    def structure_map(self, data):
        """Return a :class:`.StructureMap` of the model system mapped to the specified `data`

        Parameters
        ----------
        data : Optional[array_like]
            Data array to map to site positions.

        Returns
        -------
        :class:`~pybinding.results.StructureMap`
        """
        return self.system.with_data(data)

    def tokwant(self):
        """Convert this model into `kwant <http://kwant-project.org/>`_ format (finalized)

        This is intended for compatibility with the kwant package: http://kwant-project.org/.

        Returns
        -------
        kwant.system.System
            Finalized system which can be used with kwant compute functions.
        """
        from .support.kwant import tokwant
        return tokwant(self)

    @property
    def system(self) -> System:
        """Structural data like site positions and hoppings, see :class:`.System` for details"""
        return System(super().system)

    @property
    def hamiltonian(self) -> csr_matrix:
        """Hamiltonian sparse matrix in the :class:`.scipy.sparse.csr_matrix` format"""
        return super().hamiltonian

    @property
    def lattice(self) -> Lattice:
        """:class:`.Lattice` specification"""
        return self._lattice
    
    @property
    def leads(self):
        """List of :class:`.Lead` objects"""
        return Leads(super().leads)

    @property
    def shape(self):
        """:class:`.Polygon` or :class:`.FreeformShape` object"""
        return self._shape

    @property
    def modifiers(self) -> list:
        """List of all modifiers applied to this model"""
        return (self.state_modifiers + self.position_modifiers +
                self.onsite_modifiers + self.hopping_modifiers)

    @property
    def onsite_map(self) -> results.StructureMap:
        """:class:`.StructureMap` of the onsite energy"""
        return self.structure_map(np.real(self.hamiltonian.diagonal()))

    def plot(self, num_periods=1, lead_length=6, axes='xy', **kwargs):
        """Plot the structure of the model: sites, hoppings, boundaries and leads

        Parameters
        ----------
        num_periods : int
            Number of times to repeat the periodic boundaries.
        lead_length : int
            Number of times to repeat the lead structure.
        axes : str
            The spatial axes to plot. E.g. 'xy', 'yz', etc.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        kwargs['add_margin'] = False
        self.system.plot(num_periods, axes=axes, **kwargs)
        for lead in self.leads:
            lead.plot(lead_length, axes=axes, **kwargs)
        decorate_structure_plot(axes=axes)
