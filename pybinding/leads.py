"""Lead interface for transport calculations

The only way to create leads is using the :meth:`.Model.attach_lead` method.
The classes represented here are the final product of that process, listed
in :attr:`.Model.leads`.
"""
import numpy as np
from math import pi
from scipy.sparse import csr_matrix

from . import _cpp
from . import pltutils, results
from .system import System, plot_sites, plot_hoppings, get_structure_props, decorate_structure_plot

__all__ = ['Lead']


def _center(pos, shift):
    """Return the 2D center position of `pos + shift`"""
    x = np.concatenate((pos[0], pos[0] + shift[0]))
    y = np.concatenate((pos[1], pos[1] + shift[1]))
    return (x.max() + x.min()) / 2, (y.max() + y.min()) / 2


class Lead:
    """Describes a single lead connected to a :class:`.Model`

    Leads can only be created using :meth:`.Model.attach_lead`
    and accessed using :attr:`.Model.leads`.
    """
    def __init__(self, impl: _cpp.Lead, index):
        self.impl = impl
        self.index = index

    @property
    def indices(self) -> np.ndarray:
        """Main system indices (1d array) to which this lead is connected"""
        return self.impl.indices

    @property
    def system(self) -> System:
        """Structural information, see :class:`.System`"""
        return System(self.impl.system)

    @property
    def h0(self) -> csr_matrix:
        """Unit cell Hamiltonian as :class:`~scipy.sparse.csr_matrix`"""
        return self.impl.h0

    @property
    def h1(self) -> csr_matrix:
        """Hamiltonian which connects who unit cells, :class:`~scipy.sparse.csr_matrix`"""
        return self.impl.h1

    def calc_bands(self, start=-pi, end=pi, step=0.05):
        """Calculate the band structure of an infinite lead

        Parameters
        ----------
        start, end : float
            Points in reciprocal space which form the path for the band calculation.
        step : float
            Calculation step length in reciprocal space units. Lower `step` values
            will return more detailed results.

        Returns
        -------
        :class:`~pybinding.results.Bands`
        """
        from scipy.linalg import eigh

        h0 = self.h0.todense()
        h1 = self.h1.todense()
        h1t = np.conj(h1.T)

        def eigenvalues(k):
            h = h0 + h1 * np.exp(1j * k) + h1t * np.exp(-1j * k)
            return eigh(h, eigvals_only=True)

        k_path = results.make_path(start, end, step=step).flatten()
        bands = [eigenvalues(k) for k in k_path]
        return results.Bands([start, end], k_path, np.vstack(bands))

    def plot(self, site_radius=0.025, hopping_width=1.0, lead_length=6, axes='xy', **kwargs):
        """Plot the sites, hoppings and periodic boundaries of the lead

        Parameters
        ----------
        site_radius : float
            Radius (in data units) of the circle representing a lattice site.
        hopping_width : float
            Width (in figure units) of the hopping lines.
        lead_length : int
            Number of times to repeat the lead's periodic boundaries.
        axes : str
            The spatial axes to plot. E.g. 'xy', 'yz', etc.
        **kwargs
            Site, hopping and boundary properties: to be forwarded to their respective plots.
        """
        pos = self.system.positions
        sub = self.system.sublattices
        inner_hoppings = self.system.hoppings.tocoo()
        boundary = self.system.boundaries[0]
        outer_hoppings = boundary.hoppings.tocoo()

        props = get_structure_props(axes, **kwargs)
        blend_gradient = np.linspace(0.5, 0.1, lead_length)
        for i, blend in enumerate(blend_gradient):
            offset = i * boundary.shift
            plot_sites(pos, sub, site_radius, offset, blend, **props['site'])
            plot_hoppings(pos, inner_hoppings, hopping_width, offset, blend, **props['hopping'])
            plot_hoppings(pos, outer_hoppings, hopping_width * 1.6, offset - boundary.shift, blend,
                          boundary=(1, boundary.shift), **props['boundary'])

        label_pos = _center(pos, lead_length * boundary.shift * 1.5)
        pltutils.annotate_box("lead {}".format(self.index), label_pos, bbox=dict(alpha=0.7))
        decorate_structure_plot(axes)

    def plot_bands(self, start=-pi, end=pi, step=0.05, **kwargs):
        """Plot the band structure of an infinite lead

        Parameters
        ----------
        start, end : float
            Points in reciprocal space which form the path for the band calculation.
        step : float
            Calculation step length in reciprocal space units. Lower `step` values
            will return more detailed results.
        **kwargs
            Forwarded to :meth:`.Bands.plot`.
        """
        bands = self.calc_bands(start, end, step)
        bands.plot(**kwargs)


class Leads:
    def __init__(self, impl: _cpp.Leads):
        self.impl = impl

    def __getitem__(self, index):
        return Lead(self.impl[index], index)

    def __len__(self):
        return len(self.impl)
