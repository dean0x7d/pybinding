"""Crystal lattice specification"""
import itertools
import warnings
from copy import deepcopy
from math import pi, atan2, sqrt

import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import x_pi, with_defaults, rotate_axes
from .support.deprecated import LoudDeprecationWarning

__all__ = ['Lattice']


class Lattice:
    """Unit cell of a Bravais lattice, the basic building block of a tight-binding model

    This class describes the primitive vectors, positions of sublattice sites and hopping
    parameters which connect those sites. All of this structural information is used to
    build up a larger system by translation.

    A few prebuilt lattices are available in the :doc:`/materials/index`.

    Parameters
    ----------
    a1, a2, a3 : array_like
        Primitive vectors of a Bravais lattice. A valid lattice must have at least
        one primitive vector (`a1`), thus forming a simple 1-dimensional lattice.
        If `a2` is also specified, a 2D lattice is created. Passing values for all
        three vectors will create a 3D lattice.
    """
    def __init__(self, a1, a2=None, a3=None):
        vectors = (v for v in (a1, a2, a3) if v is not None)
        self.impl = _cpp.Lattice(*vectors)

    @classmethod
    def from_impl(cls, impl: _cpp.Lattice) -> 'Lattice':
        lat = cls.__new__(cls)
        lat.impl = impl
        return lat

    @property
    def ndim(self) -> int:
        """The dimensionality of the lattice: number of primitive vectors"""
        return self.impl.ndim

    @property
    def nsub(self) -> int:
        """Number of sublattices"""
        return self.impl.nsub

    @property
    def nhop(self) -> int:
        """Number of hopping families"""
        return self.impl.nhop

    @property
    def vectors(self) -> list:
        """Primitive lattice vectors"""
        return self.impl.vectors

    @property
    def sublattices(self) -> dict:
        """Dict of names and :class:`~_pybinding.Sublattice`"""
        return self.impl.sublattices

    @property
    def hoppings(self) -> dict:
        """Dict of names and :class:`~_pybinding.HoppingFamily`"""
        return self.impl.hoppings

    @property
    def offset(self):
        """Global lattice offset: sublattice offsets are defined relative to this

        It must be within half the length of a primitive lattice vector."""
        return self.impl.offset

    @offset.setter
    def offset(self, value):
        self.impl.offset = value

    @property
    def min_neighbors(self):
        """Minimum number of neighbours required at each lattice site

        When constructing a finite-sized system, lattice sites with less neighbors
        than this minimum will be considered as "dangling" and they will be removed."""
        return self.impl.min_neighbors

    @min_neighbors.setter
    def min_neighbors(self, value):
        self.impl.min_neighbors = value

    def __getitem__(self, name):
        warnings.warn("Use the sublattice name directly instead",
                      LoudDeprecationWarning, stacklevel=2)
        return name

    def __call__(self, name):
        warnings.warn("Use the hopping name directly instead",
                      LoudDeprecationWarning, stacklevel=2)
        return name

    def register_hopping_energies(self, mapping):
        """Register a mapping of user-friendly names to hopping energies

        Parameters
        ----------
        mapping : dict
            Keys are user-friendly hopping names and values are the numeric values
            of the hopping energy.
        """
        for name, energy in sorted(mapping.items(), key=lambda item: item[0]):
            self.impl.register_hopping_energy(name, energy)

    def add_one_sublattice(self, name, position, onsite_energy=0.0, alias=""):
        """Add a new sublattice

        Parameters
        ----------
        name : str
            User-friendly identifier. The unique sublattice ID can later be accessed
            via this sublattice name as `lattice[sublattice_name]`.
        position : array_like
            Cartesian position with respect to the origin.
        onsite_energy : float
            Onsite energy to be applied only to sites of this sublattice.
        alias : str
            Deprecated: Use :meth:`add_one_alias` instead.
        """
        if alias:
            warnings.warn("Use Lattice.add_aliases() instead",
                          LoudDeprecationWarning, stacklevel=2)
            self.add_one_alias(name, alias, position)
        else:
            self.impl.add_sublattice(name, position, np.asarray(onsite_energy))

    def add_sublattices(self, *sublattices):
        """Add multiple new sublattices

        Parameters
        ----------
        *sublattices
            Each element should be a tuple containing the arguments for
            a `add_one_sublattice()` method call. See example.

        Examples
        --------
        These three calls::

            lattice.add_one_sublattice('a', [0, 0], 0.5)
            lattice.add_one_sublattice('b', [0, 1], 0.0)
            lattice.add_one_sublattice('c', [1, 0], 0.3)

        Can be replaced with a single call to::

            lattice.add_sublattices(
                ('a', [0, 0], 0.5),
                ('b', [0, 1], 0.0),
                ('c', [1, 0], 0.3)
            )
        """
        for sub in sublattices:
            self.add_one_sublattice(*sub)

    def add_one_alias(self, name, original, position):
        """Add a sublattice alias - useful for creating supercells

        Create a new sublattice called `name` with the same properties as `original`
        (same onsite energy) but with at a different `position`. The new `name` is
        only used during lattice construction and the `original` will be used for the
        final system and Hamiltonian. This is useful when defining a supercell which
        contains multiple sites of one sublattice family at different positions.

        Parameters
        ----------
        name : str
           User-friendly identifier of the alias.
        original : str
           Name of the original sublattice. It must already exist.
        position : array_like
           Cartesian position with respect to the origin. Usually different than the original.
        """
        self.impl.add_alias(name, original, position)

    def add_aliases(self, *aliases):
        """Add multiple new aliases

        Parameters
        ----------
        *aliases
            Each element should be a tuple containing the arguments for
            :meth:`add_one_alias`. Works just like :meth:`add_sublattices`.
        """
        for alias in aliases:
            self.add_one_alias(*alias)

    def add_one_hopping(self, relative_index, from_sub, to_sub, hop_name_or_energy):
        """Add a new hopping

        For each new hopping, its Hermitian conjugate is added automatically. Doing so
        manually, i.e. adding a hopping which is the Hermitian conjugate of an existing
        one, will result in an exception being raised.

        Parameters
        ----------
        relative_index : array_like of int
            Difference of the indices of the source and destination unit cells.
        from_sub : str
            Name of the sublattice in the source unit cell.
        to_sub : str
            Name of the sublattice in the destination unit cell.
        hop_name_or_energy : float or str
            The numeric value of the hopping energy or the name of a previously
            registered hopping.
        """
        self.impl.add_hopping(relative_index, from_sub, to_sub, hop_name_or_energy)

    def add_hoppings(self, *hoppings):
        """Add multiple new hoppings

        Parameters
        ----------
        *hoppings
            Each element should be a tuple containing the arguments for
            a `add_one_hopping()` method call. See example.

        Examples
        --------
        These three calls::

            lattice.add_one_hopping([0, 0], 'a', 'b', 0.8)
            lattice.add_one_hopping([0, 1], 'a', 'a', 0.3)
            lattice.add_one_hopping([1, 1], 'a', 'b', 0.8)

        Can be replaced with a single call to::

            lattice.add_hoppings(
                ([0, 0], 'a', 'b', 0.8),
                ([0, 1], 'a', 'a', 0.3),
                ([1, 1], 'a', 'b', 0.8),
            )
        """
        for hop in hoppings:
            self.add_one_hopping(*hop)

    def with_offset(self, position):
        """Return a copy of this lattice with a different offset

        It must be within half the length of a primitive lattice vector

        Parameters
        ----------
        position : array_like
            Cartesian offset in the same length unit as the lattice vectors.

        Returns
        -------
        :class:`Lattice`
        """
        cp = deepcopy(self)
        cp.offset = position
        return cp

    def with_min_neighbors(self, number):
        """Return a copy of this lattice with a different minimum neighbor count

        Parameters
        ----------
        number : int
            The minimum number of neighbors.

        Returns
        -------
        :class:`Lattice`
        """
        cp = deepcopy(self)
        cp.min_neighbors = number
        return cp

    def reciprocal_vectors(self):
        """Calculate the reciprocal space lattice vectors

        Returns
        -------
        list

        Examples
        --------
        >>> lat = Lattice(a1=[0, 1], a2=[0.5, 0.5])
        >>> np.allclose(lat.reciprocal_vectors(), [[4*pi, 0, 0], [-2*pi, 2*pi, 0]])
        True
        """
        n = self.ndim
        mat = np.column_stack(self.vectors)[:n]
        mat = 2 * pi * np.linalg.inv(mat).T
        mat = np.vstack([mat, np.zeros(shape=(3 - n, n))])
        return [v.squeeze() for v in reversed(np.hsplit(mat, n))]

    def brillouin_zone(self):
        """Return a list of vertices which form the Brillouin zone (1D and 2D only)

        Returns
        -------
        List[array_like]

        Examples
        --------
        >>> lat_1d = Lattice(a1=1)
        >>> np.allclose(lat_1d.brillouin_zone(), [-pi, pi])
        True
        >>> lat_2d = Lattice(a1=[0, 1], a2=[0.5, 0.5])
        >>> np.allclose(lat_2d.brillouin_zone(), [[0, -2*pi], [2*pi, 0], [0, 2*pi], [-2*pi, 0]])
        True
        """
        from scipy.spatial import Voronoi

        if self.ndim == 1:
            v1, = self.reciprocal_vectors()
            l = np.linalg.norm(v1)
            return [-l/2, l/2]
        elif self.ndim == 2:
            # The closest reciprocal lattice points are combinations of the primitive vectors
            vectors = self.reciprocal_vectors()
            points = [sum(n * v for n, v in zip(ns, vectors))
                      for ns in itertools.product([-1, 0, 1], repeat=self.ndim)]

            # Voronoi doesn't like trailing zeros in coordinates
            vor = Voronoi([p[:self.ndim] for p in points])

            # See scipy's Voronoi documentation for details (-1 indicates infinity)
            finite_regions = [r for r in vor.regions if len(r) != 0 and -1 not in r]
            assert len(finite_regions) == 1
            bz_vertices = [vor.vertices[i] for i in finite_regions[0]]

            # sort counter-clockwise
            return sorted(bz_vertices, key=lambda v: atan2(v[1], v[0]))
        else:
            raise RuntimeError("3D Brillouin zones are not currently supported")

    @staticmethod
    def _plot_vectors(vectors, position=(0, 0), name="a", scale=1.0,
                      head_width=0.08, head_length=0.2):
        vnorm = np.average([np.linalg.norm(v) for v in vectors]) * scale
        for i, vector in enumerate(vectors):
            v2d = np.array(vector[:2]) * scale
            if np.allclose(v2d, [0, 0]):
                continue  # nonzero only in z dimension, but the plot is 2D

            plt.arrow(position[0], position[1], *v2d, color='black', length_includes_head=True,
                      head_width=vnorm * head_width, head_length=vnorm * head_length)
            pltutils.annotate_box(r"${}_{}$".format(name, i+1), position[:2] + v2d / 2,
                                  fontsize='large', bbox=dict(lw=0, alpha=0.6))

    def plot_vectors(self, position, scale=1.0):
        """Plot lattice vectors in the xy plane

        Parameters
        ----------
        position : array_like
            Cartesian position to be used as the origin for the vectors.
        scale : float
            Multiply the length of the vectors by this number.
        """
        self._plot_vectors(self.vectors, position, scale=scale)

    def _visible_sublattices(self, axes):
        """Return the sublattices which are visible when viewed top-down in the `axes` plane"""
        idx = list(rotate_axes([0, 1, 2], axes))
        xy_idx, z_idx = idx[:2], idx[2]

        sorted_subs = sorted(self.sublattices.items(), reverse=True,
                             key=lambda pair: pair[1].position[z_idx])
        result = dict()
        seen_positions = set()
        for name, sub in sorted_subs:
            p = tuple(sub.position[xy_idx])
            if p not in seen_positions:
                seen_positions.add(p)
                result[name] = sub
        return result

    def site_radius_for_plot(self, max_fraction=0.33):
        """Return a good estimate for the lattice site radius for plotting

        Calculated heuristically base on the length (1D) or area (2D) of the unit cell.
        In order to prevent overlap between sites, if the computed radius is too large,
        it will be clamped to a fraction of the shortest inter-atomic spacing.

        Parameters
        ----------
        max_fraction : float
            Set the upper limit of the calculated radius as this fraction of the
            shortest inter-atomic spacing in the lattice unit cell. Should be less
            than 0.5 to avoid overlap between neighboring lattice sites.

        Returns
        -------
        float
        """
        def heuristic_radius(lattice):
            """The `magic` numbers were picked base on what looks nice in figures"""
            if lattice.ndim == 1:
                magic = 0.12
                return magic * np.linalg.norm(lattice.vectors[0])
            elif lattice.ndim == 2:
                v1, v2 = lattice.vectors
                unit_cell_area = np.linalg.norm(np.cross(v1, v2))

                num_visible = len(self._visible_sublattices("xy"))
                site_area = unit_cell_area / num_visible

                if (lattice.ndim / num_visible).is_integer():
                    magic = 0.35  # single layer or nicely stacked layers
                else:
                    magic = 0.42  # staggered layers

                return magic * sqrt(site_area / pi)
            else:
                magic = 0.18
                average_vec_length = sum(np.linalg.norm(v) for v in lattice.vectors) / 3
                return magic * average_vec_length

        def shortest_site_spacing(lattice):
            from scipy.spatial.distance import pdist

            distances = pdist([s.position for s in lattice.sublattices.values()])
            distances = distances[distances > 0]

            if np.any(distances):
                return np.min(distances)
            else:
                vector_lengths = [np.linalg.norm(v) for v in lattice.vectors]
                return np.min(vector_lengths)

        r1 = heuristic_radius(self)
        r2 = max_fraction * shortest_site_spacing(self)
        return min(r1, r2)

    def plot(self, axes="xy", vector_position="center", **kwargs):
        """Illustrate the lattice by plotting the primitive cell and its nearest neighbors

        Parameters
        ----------
        axes : str
            The spatial axes to plot. E.g. 'xy', 'yz', etc.
        vector_position : array_like or 'center'
            Cartesian position to be used as the origin for the lattice vectors.
            By default the origin is placed in the center of the primitive cell.
        **kwargs
            Forwarded to `System.plot()`.
        """
        from .model import Model
        from .shape import translational_symmetry

        # reuse model plotting code (kind of meta)
        model = Model(self, translational_symmetry())
        model.system.plot(**with_defaults(kwargs, hopping=dict(color='#777777', width=1),
                                          axes=axes))

        # by default, plot the lattice vectors from the center of the unit cell
        vectors = [np.array(rotate_axes(v, axes)) for v in self.vectors]
        sub_center = sum(s.position for s in self.sublattices.values()) / self.nsub
        sub_center = rotate_axes(sub_center, axes)
        if vector_position is not None:
            vector_position = sub_center if vector_position == "center" else vector_position
            self._plot_vectors(vectors, vector_position)

        # annotate sublattice names
        for name, sub in self._visible_sublattices(axes).items():
            pltutils.annotate_box(name, xy=rotate_axes(sub.position, axes)[:2],
                                  bbox=dict(boxstyle="circle,pad=0.3", alpha=0.2, lw=0))

        # collect relative indices where annotations should be drawn
        relative_indices = []
        for hopping_family in self.hoppings.values():
            for term in hopping_family.terms:
                if tuple(term.relative_index[:2]) == (0, 0):
                    continue  # skip the original cell
                relative_indices.append(term.relative_index)
                relative_indices.append(-term.relative_index)

        # 3D distance (in length units) of the neighboring cell from the original
        offsets = [sum(r * v for r, v in zip(ri, self.vectors)) for ri in relative_indices]
        offsets = [np.array(rotate_axes(p, axes)) for p in offsets]

        # annotate neighboring cell indices
        for relative_index, offset in zip(relative_indices, offsets):
            text = "[" + ", ".join(map(str, relative_index[:self.ndim])) + "]"

            # align the text so that it goes away from the original cell
            ha, va = pltutils.align(*(-offset[:2]))
            pltutils.annotate_box(text, xy=(sub_center[:2] + offset[:2]) * 1.05,
                                  ha=ha, va=va, clip_on=True, bbox=dict(lw=0))

        # ensure there is some padding around the lattice
        offsets += [(0, 0, 0)]
        points = [n * v + o for n in (-0.5, 0.5) for v in vectors for o in offsets]
        x, y, _ = zip(*points)
        pltutils.set_min_axis_length(abs(max(x) - min(x)), 'x')
        pltutils.set_min_axis_length(abs(max(y) - min(y)), 'y')
        pltutils.add_margin()

    def plot_brillouin_zone(self, decorate=True, **kwargs):
        """Plot the Brillouin zone and reciprocal lattice vectors

        Parameters
        ----------
        decorate : bool
            Label the vertices of the Brillouin zone and show the reciprocal vectors
        **kwargs
            Forwarded to `plt.plot()`.
        """
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel(r"$k_x (nm^{-1})$")

        vertices = self.brillouin_zone()
        default_color = pltutils.get_palette("Set1")[0]

        if self.ndim == 1:
            x1, x2 = vertices
            y = x2 / 10
            plt.plot([x1, x2], [y, y], **with_defaults(kwargs, color=default_color))

            ticks = [x1, 0, x2]
            plt.xticks(ticks, [x_pi(t) for t in ticks])

            plt.ylim(0, 2 * y)
            plt.yticks([])
            ax.spines['left'].set_visible(False)
        else:
            ax.add_patch(plt.Polygon(
                vertices, **with_defaults(kwargs, fill=False, color=default_color)
            ))

            if decorate:
                self._plot_vectors(self.reciprocal_vectors(), name="b",
                                   head_width=0.05, head_length=0.12)

                for vertex in vertices:
                    text = "[" + ", ".join(map(x_pi, vertex)) + "]"
                    # align the text so that it goes away from the origin
                    ha, va = pltutils.align(*(-vertex))
                    pltutils.annotate_box(text, vertex * 1.05, ha=ha, va=va, bbox=dict(lw=0))

            x, y = zip(*vertices)
            pltutils.set_min_axis_length(abs(max(x) - min(x)) * 2, 'x')
            pltutils.set_min_axis_length(abs(max(y) - min(y)) * 2, 'y')
            ax.set_ylabel(r"$k_y (nm^{-1})$")

        pltutils.despine(trim=True)
