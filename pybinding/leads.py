"""Lead interface for scattering models

The only way to create leads is using the :meth:`.Model.attach_lead` method.
The classes represented here are the final product of that process, listed
in :attr:`.Model.leads`.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import csr_matrix

from . import _cpp
from . import pltutils, results
from .system import (System, plot_sites, plot_hoppings, structure_plot_properties,
                     decorate_structure_plot)

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
        return results.Bands(k_path, np.vstack(bands))

    def plot(self, lead_length=6, **kwargs):
        """Plot the sites, hoppings and periodic boundaries of the lead

        Parameters
        ----------
        lead_length : int
            Number of times to repeat the lead's periodic boundaries.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        pos = self.system.positions
        sub = self.system.sublattices
        inner_hoppings = self.system.hoppings.tocoo()
        boundary = self.system.boundaries[0]
        outer_hoppings = boundary.hoppings.tocoo()

        props = structure_plot_properties(**kwargs)
        props['site'].setdefault('radius', self.system.lattice.site_radius_for_plot())

        blend_gradient = np.linspace(0.5, 0.1, lead_length)
        for i, blend in enumerate(blend_gradient):
            offset = i * boundary.shift
            plot_sites(pos, sub, offset=offset, blend=blend, **props['site'])
            plot_hoppings(pos, inner_hoppings, offset=offset, blend=blend, **props['hopping'])
            plot_hoppings(pos, outer_hoppings, offset=offset - boundary.shift, blend=blend,
                          boundary=(1, boundary.shift), **props['boundary'])

        label_pos = _center(pos, lead_length * boundary.shift * 1.5)
        pltutils.annotate_box("lead {}".format(self.index), label_pos, bbox=dict(alpha=0.7))

        decorate_structure_plot(**props)

    def plot_contact(self, line_width=1.6, arrow_length=0.5,
                     shade_width=0.3, shade_color='#d40a0c'):
        """Plot the shape and direction of the lead contact region

        Parameters
        ----------
        line_width : float
            Width of the line representing the lead contact.
        arrow_length : float
            Size of the direction arrow as a fraction of the contact line length.
        shade_width : float
            Width of the shaded area as a fraction of the arrow length.
        shade_color : str
            Color of the shaded area.
        """
        lead_spec = self.impl.spec
        vectors = self.impl.system.lattice.vectors
        if len(lead_spec.shape.vertices) != 2 or len(vectors) != 2:
            raise RuntimeError("This only works for 2D systems")

        # contact line vertices
        a, b = (v[:2] for v in lead_spec.shape.vertices)

        def plot_contact_line():
            # Not using plt.plot() because it would reset axis limits
            plt.gca().add_patch(plt.Polygon([a, b], color='black', lw=line_width))

        def rescale_lattice_vector(vec):
            line_length = np.linalg.norm(a - b)
            scale = arrow_length * line_length / np.linalg.norm(vec)
            return vec[:2] * scale

        def plot_arrow(xy, vec, spec, head_width=0.08, head_length=0.2):
            vnorm = np.linalg.norm(vec)
            plt.arrow(xy[0], xy[1], *vec, color='black', alpha=0.9, length_includes_head=True,
                      head_width=vnorm * head_width, head_length=vnorm * head_length)
            label = r"${}a_{}$".format("-" if spec.sign < 0 else "", spec.axis + 1)
            pltutils.annotate_box(label, xy + vec / 5, fontsize='large',
                                  bbox=dict(lw=0, alpha=0.6))

        def plot_polygon(w):
            plt.gca().add_patch(plt.Polygon([a - w, a + w, b + w, b - w],
                                            color=shade_color, alpha=0.25, lw=0))

        plot_contact_line()
        v = rescale_lattice_vector(vectors[lead_spec.axis] * lead_spec.sign)
        plot_arrow(xy=(a + b) / 2, vec=v, spec=lead_spec)
        plot_polygon(w=shade_width * v)

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
        plt.title("lead {}".format(self.index))


class Leads:
    def __init__(self, impl: _cpp.Leads):
        self.impl = impl

    def __getitem__(self, index):
        return Lead(self.impl[index], index)

    def __len__(self):
        return len(self.impl)
