from math import pi, atan2
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from . import _cpp
from . import pltutils
from .utils import x_pi, with_defaults
from .support.pickle import pickleable

__all__ = ['Lattice', 'make_lattice']


@pickleable(props='sublattices hopping_energies min_neighbors')
class Lattice(_cpp.Lattice):
    """Holds the specification of a tight-binding lattice

    Parameters
    ----------
    a1, a2, a3 : array_like
        Primitive vectors of a Bravais lattice. A valid lattice must have at least
        one primitive vector (`a1`), thus forming a simple 1-dimensional lattice.
        If `a2` is also specified, a 2D lattice is created. Passing values for all
        three vectors will create a 3D lattice.

    Attributes
    ----------
    sublattice_ids : dict
        Map from sublattice names to unique IDs. Shortcut: `lattice[sublattice_name]`
    hopping_ids : dict
        Map from hopping names to unique IDs. Shortcut: `lattice(hopping_name)`
    """
    def __init__(self, a1, a2=None, a3=None):
        vectors = (np.atleast_1d(v) for v in (a1, a2, a3) if v is not None)
        super().__init__(*vectors)
        self.sublattice_ids = OrderedDict()
        self.hopping_ids = {}

    def __getitem__(self, name):
        """Get sublattice ID from its user-friendly `name`"""
        if isinstance(name, str):
            if name not in self.sublattice_ids:
                raise KeyError("There is no sublattice named '{}'".format(name))
            return self.sublattice_ids[name]
        else:  # an ID was given instead of a name, verify it
            sub_id = name
            if sub_id not in self.sublattice_ids.values():
                raise KeyError("There is no sublattice with ID = {}".format(sub_id))
            return sub_id

    def __call__(self, name):
        """Get the hopping ID from its user-friendly `name`"""
        if name not in self.hopping_ids:
            raise KeyError("There is no hopping named '{}'".format(name))
        return self.hopping_ids[name]

    @property
    def ndim(self):
        """The dimensionality of the lattice: number of primitive vectors"""
        return len(self.vectors)

    def register_hopping_energies(self, mapping):
        """Register a mapping of user-friendly names to hopping energies

        Parameters
        ----------
        mapping : dict
            Keys are user-friendly hopping names and values are the numeric values
            of the hopping energy.
        """
        for name, energy in sorted(mapping.items(), key=lambda item: item[0]):
            if name in self.hopping_ids:
                raise KeyError("Hopping '{}' already exists".format(name))
            self.hopping_ids[name] = super()._register_hopping_energy(energy)

    def add_one_sublattice(self, name, offset, onsite_energy=0.0, alias=None):
        """Add a new sublattice

        Parameters
        ----------
        name : str
            User-friendly identifier. The unique sublattice ID can later be accessed
            via this sublattice name as `lattice[sublattice_name]`.
        offset : array_like
            Cartesian position with respect to the origin.
        onsite_energy : float, optional
            Onsite energy to be applied only to sites of this sublattice.
        alias : str, optional
            Given the name of a previously defined sublattice, the new sublattice
            is created as an alias for the old one. This is useful when defining
            a supercell which contains multiple sites of one sublattice family at
            different positions.
        """
        if name in self.sublattice_ids:
            raise KeyError("Sublattice '{}' already exists".format(name))

        offset = np.atleast_1d(offset)
        alias = self.__getitem__(alias) if alias is not None else -1
        self.sublattice_ids[name] = super()._add_sublattice(offset, onsite_energy, alias)

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

    def add_one_hopping(self, relative_index, from_sublattice, to_sublattice, energy):
        """Add a new hopping

        For each new hopping, its Hermitian conjugate is added automatically. Doing so
        manually, i.e. adding a hopping which is the Hermitian conjugate of an existing
        one, will result in an exception being raised.

        Parameters
        ----------
        relative_index : array_like of int
            Difference of the indices of the source and destination unit cells.
        from_sublattice : str
            Name of the sublattice in the source unit cell.
        to_sublattice : str
            Name of the sublattice in the destination unit cell.
        energy : float or str
            The numeric value of the hopping energy or the name of a previously
            registered hopping.
        """
        relative_index = np.atleast_1d(relative_index)
        from_sub, to_sub = map(self.__getitem__, (from_sublattice, to_sublattice))

        if energy in self.hopping_ids:
            super()._add_registered_hopping(relative_index, from_sub, to_sub,
                                            self.hopping_ids[energy])
        elif not isinstance(energy, str):
            super()._add_hopping(relative_index, from_sub, to_sub, energy)
        else:
            raise KeyError("There is no hopping named '{}'".format(energy))

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

    def add_hopping_matrices(self, *pairs):
        """Add hoppings in the matrix format

        Parameters
        ----------
        *pairs
            Each element is a tuple of `relative_index` and `hopping_matrix`.
        """
        for relative_index, matrix in pairs:
            for (from_sub, to_sub), hopping_energy in np.ndenumerate(matrix):
                if hopping_energy == 0:
                    continue
                # only consider lower triangle values of the relative_index==(0, 0) matrix
                # the upper triangle is implied via Hermitian conjugation
                if all(v == 0 for v in relative_index) and from_sub >= to_sub:
                    continue

                self.add_one_hopping(relative_index, from_sub, to_sub, hopping_energy)

    def reciprocal_vectors(self) -> list:
        """Calculate the reciprocal space lattice vectors

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
        """Return a list of vertices which form the Brillouin zone (2D only)

        Returns
        -------
        list of array_like

        Examples
        --------
        >>> lat = Lattice(a1=[0, 1], a2=[0.5, 0.5])
        >>> np.allclose(lat.brillouin_zone(), [[-2*pi, 0], [0, -2*pi], [2*pi, 0], [0, 2*pi]])
        True
        """
        def _perp(vec):
            """Make 'a*x + b*y = c' line which is perpendicular to the vector"""
            v_rot90 = [-vec[1], vec[0], 0]
            x1, y1, _ = 0.5 * (vec + v_rot90)
            x2, y2, _ = 0.5 * (vec - v_rot90)

            a, b = y1 - y2, x2 - x1
            c = x2 * y1 - x1 * y2
            return a, b, c

        def _intersection(line1, line2):
            """Return (x, y) coordinates of the intersection via Cramer's rule"""
            d = np.linalg.det([line1[:2], line2[:2]])
            dx = np.linalg.det([line1[1:], line2[1:]])
            dy = np.linalg.det([line1[::2], line2[::2]])
            return np.array([dx / d, dy / d]) if d != 0 else None

        if self.ndim == 1:
            v1, = self.reciprocal_vectors()
            return [v1[0], -v1[0]]
        elif self.ndim == 2:
            # list all combinations of primitive reciprocal vectors
            v1, v2 = self.reciprocal_vectors()
            vectors = [n1 * v1 + n2 * v2 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) if n1 != -n2]

            vertices = []
            for v1 in vectors:
                intersections = [_intersection(_perp(v1), _perp(v2)) for v2 in vectors]
                intersections = filter(lambda v: v is not None, intersections)

                # keep only the two closest to (0, 0)
                vertices += sorted(intersections, key=lambda v: np.linalg.norm(v))[:2]

            unique_vertices = []
            for vertex in vertices:
                if not any(np.allclose(vertex, v0) for v0 in unique_vertices):
                    unique_vertices.append(vertex)

            # sort counter-clockwise
            return sorted(unique_vertices, key=lambda v: atan2(v[1], v[0]))
        else:
            raise RuntimeError("3D Brillouin zones are not currently supported")

    @staticmethod
    def _plot_vectors(vectors, position=(0, 0), name="a", head_width=0.08, head_length=0.2):
        vnorm = np.average([np.linalg.norm(v) for v in vectors])
        for i, vector in enumerate(vectors):
            v2d = vector[:2]
            if np.allclose(v2d, [0, 0]):
                continue  # nonzero only in z dimension, but the plot is 2D

            plt.arrow(position[0], position[1], *v2d, color='black', alpha=0.8,
                      head_width=vnorm * head_width, head_length=vnorm * head_length,
                      length_includes_head=True)
            pltutils.annotate_box(r"${}_{}$".format(name, i+1), position[:2] + v2d / 2,
                                  fontsize='large', bbox=dict(lw=0, alpha=0.6))

    def plot_vectors(self, position):
        """Plot lattice vectors in the xy plane

        Parameters
        ----------
        position : array_like
            Cartesian position to be used as the origin for the vectors.
        """
        self._plot_vectors(self.vectors, position)

    def plot(self, vector_position='center', **kwargs):
        """Illustrate the lattice by plotting the primitive cell and its nearest neighbors

        Parameters
        ----------
        vector_position : array_like or 'center'
            Cartesian position to be used as the origin for the lattice vectors.
            By default the origin is placed in the center of the primitive cell.
        **kwargs
            Forwarded to `System.plot()`.
        """
        import pybinding as pb
        # reuse model plotting code (kind of meta)
        model = pb.Model(self, pb.translational_symmetry())
        model.system.plot(**with_defaults(kwargs, hopping_props=dict(colors='#777777')))

        # by default, plot the lattice vectors from the center of the unit cell
        sub_center = sum(s.offset for s in self.sublattices) / len(self.sublattices)
        if vector_position is not None:
            self.plot_vectors(sub_center if vector_position == 'center' else vector_position)

        # annotate sublattice names
        sub_names = list(self.sublattice_ids.keys())
        for sub in self.sublattices:
            pltutils.annotate_box(sub_names[sub.alias], xy=sub.offset[:2], bbox=dict(lw=0))

        # annotate neighboring cell indices
        offsets = [(0, 0, 0)]
        for sub in self.sublattices:
            for hop in sub.hoppings:
                if tuple(hop.relative_index[:2]) == (0, 0):
                    continue  # skip the original cell

                # offset of the neighboring cell from the original
                offset = sum(r * v for r, v in zip(hop.relative_index, self.vectors))
                offsets.append(offset)

                text = "[" + ", ".join(map(str, hop.relative_index[:self.ndim])) + "]"

                # align the text so that it goes away from the original cell
                ha, va = pltutils.align(*(-offset[:2]))
                pltutils.annotate_box(text, xy=(sub_center[:2] + offset[:2]) * 1.05,
                                      ha=ha, va=va, clip_on=True, bbox=dict(lw=0))

        # ensure there is some padding around the lattice
        points = [n * v + o for n in (-0.5, 0.5) for v in self.vectors for o in offsets]
        x, y, _ = zip(*points)
        pltutils.set_min_axis_length(abs(max(x) - min(x)), 'x')
        pltutils.set_min_axis_length(abs(max(y) - min(y)), 'y')

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


def make_lattice(vectors, sublattices, hoppings, min_neighbors=1):
    lat = Lattice(*vectors)
    lat.add_sublattices(*sublattices)
    lat.add_hoppings(*hoppings)
    lat.min_neighbors = min_neighbors
    return lat
