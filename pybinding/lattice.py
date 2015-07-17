from math import pi, atan2
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import _pybinding
from . import pltutils
from .utils import x_pi
from .support.pickle import pickleable

__all__ = ['Lattice', 'make_lattice', 'square']


@pickleable(props='sublattices hopping_energies min_neighbors')
class Lattice(_pybinding.Lattice):
    def __init__(self, v1, v2=None, v3=None):
        super().__init__(*(v for v in [v1, v2, v3] if v is not None))
        self.sublattice_ids = OrderedDict()  # indexed by sublattice name
        self.hopping_ids = {}  # indexed by hopping name

    def __getitem__(self, key):
        """Get sublattice ID from name."""
        if isinstance(key, str):
            if key not in self.sublattice_ids:
                raise KeyError("There is no sublattice named '{}'".format(key))
            return self.sublattice_ids[key]
        else:  # an ID was given instead of a name, verify it
            if key not in self.sublattice_ids.values():
                raise KeyError("There is no sublattice with ID = {}".format(key))
            return key

    @property
    def ndim(self):
        return len(self.vectors)

    def register_hopping_energies(self, mapping: dict):
        for name, energy in mapping.items():
            if name in self.hopping_ids:
                raise KeyError("Hopping '{}' already exists".format(name))
            self.hopping_ids[name] = super()._register_hopping_energy(energy)

    def add_one_sublattice(self, name, offset, onsite_potential=0.0, alias=None):
        if name in self.sublattice_ids:
            raise KeyError("Sublattice '{}' already exists".format(name))

        alias = self.__getitem__(alias) if alias is not None else -1
        self.sublattice_ids[name] = super()._add_sublattice(offset, onsite_potential, alias)

    def add_sublattices(self, *sublattices):
        for sub in sublattices:
            self.add_one_sublattice(*sub)

    def add_one_hopping(self, relative_index, from_sublattice, to_sublattice, energy):
        from_sub, to_sub = map(self.__getitem__, (from_sublattice, to_sublattice))
        if energy in self.hopping_ids:
            super()._add_registered_hopping(relative_index, from_sub, to_sub,
                                            self.hopping_ids[energy])
        elif not isinstance(energy, str):
            super()._add_hopping(relative_index, from_sub, to_sub, energy)
        else:
            raise KeyError("There is no hopping named '{}'".format(energy))

    def add_hoppings(self, *hoppings):
        for hop in hoppings:
            self.add_one_hopping(*hop)

    def add_hopping_matrices(self, *pairs):
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
        """Calculate reciprocal space lattice vectors

        >>> lat = Lattice(v1=(0, 1), v2=(0.5, 0.5))
        >>> np.allclose(lat.reciprocal_vectors(), [[4*pi, 0, 0], [-2*pi, 2*pi, 0]])
        True
        """
        n = self.ndim
        mat = np.column_stack(self.vectors)[:n]
        mat = 2 * pi * np.linalg.inv(mat).T
        mat = np.vstack([mat, np.zeros(shape=(3 - n, n))])
        return [v.squeeze() for v in reversed(np.hsplit(mat, n))]

    def brillouin_zone(self) -> list:
        """Return a list of vertices which form the Brillouin zone (2D only)

        >>> lat = Lattice(v1=(0, 1), v2=(0.5, 0.5))
        >>> np.allclose(lat.brillouin_zone(), [[-2*pi, 0], [0, -2*pi], [2*pi, 0], [0, 2*pi]])
        True
        """
        def perp(vec):
            """Make 'a*x + b*y = c' line which is perpendicular to the vector"""
            v_rot90 = [-vec[1], vec[0], 0]
            x1, y1, _ = 0.5 * (vec + v_rot90)
            x2, y2, _ = 0.5 * (vec - v_rot90)

            a, b = y1 - y2, x2 - x1
            c = x2 * y1 - x1 * y2
            return a, b, c

        def intersection(line1, line2):
            """Return (x, y) coordinates of the intersection via Cramer's rule"""
            d = np.linalg.det([line1[:2], line2[:2]])
            dx = np.linalg.det([line1[1:], line2[1:]])
            dy = np.linalg.det([line1[::2], line2[::2]])
            return np.array([dx / d, dy / d]) if d != 0 else None

        if self.ndim == 1:
            v1, = self.reciprocal_vectors()
            return [v1[:2], -v1[:2]]
        elif self.ndim == 2:
            # list all combinations of primitive reciprocal vectors
            v1, v2 = self.reciprocal_vectors()
            vectors = [n1 * v1 + n2 * v2 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) if n1 != -n2]

            vertices = []
            for v1 in vectors:
                intersections = [intersection(perp(v1), perp(v2)) for v2 in vectors]
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
            raise RuntimeError("3D brilloun zones are not currently supported")

    @staticmethod
    def _plot_vectors(vectors, position, name="v", head_width=0.08, head_length=0.2):
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
        """Plot lattice vectors in xy plane"""
        self._plot_vectors(self.vectors, position)

    def plot(self, vector_position='center', **kwargs):
        # plot the primitive cell and it's neighbors (using a model... kind of meta)
        import pybinding as pb
        model = pb.Model(self, pb.symmetry.translational())
        model.system.plot(boundary_color=None, **kwargs)

        sub_center = sum(s.offset for s in self.sublattices) / len(self.sublattices)
        if vector_position is not None:
            self.plot_vectors(sub_center if vector_position == 'center' else vector_position)

        points = [n * v for v in self.vectors for n in (-1, 1)]  # for plot limit detection
        sub_names = list(self.sublattice_ids.keys())
        overlap = any(np.allclose(sub_center[:2], s.offset[:2]) for s in self.sublattices)

        for sub in self.sublattices:
            # annotate sublattice names
            pltutils.annotate_box(sub_names[sub.alias], xy=sub.offset[:2])

            for hop in sub.hoppings:
                # annotate neighboring cell indices
                if tuple(hop.relative_index[:2]) == (0, 0):
                    continue  # skip the original cell

                offset = sum(r * v for r, v in zip(hop.relative_index, self.vectors))
                points += (0.5 * r * v + offset for r, v in zip(hop.relative_index, self.vectors))

                mul = 1.2 if overlap else 1  # prevent text from overlapping with site
                xy = offset[:2] * mul + sub_center[:2]
                pltutils.annotate_box("{}, {}".format(*hop.relative_index[:2]), xy=xy)

        x, y, _ = zip(*points)
        pltutils.set_min_range(abs(max(x) - min(x)), 'x')
        pltutils.set_min_range(abs(max(y) - min(y)), 'y')

    def plot_brillouin_zone(self):
        vertices = self.brillouin_zone()
        x, y = zip(*vertices)
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), color=pltutils.get_palette('Set1')[0])

        self._plot_vectors(self.reciprocal_vectors(), (0, 0), name="b",
                           head_width=0.05, head_length=0.12)

        for vertex in vertices:
            ha, va = pltutils.align(*(-vertex))
            vertex_str = ", ".join(map(x_pi, vertex))
            pltutils.annotate_box("[{}]".format(vertex_str), xy=vertex * 1.05,
                                  bbox=dict(lw=0), ha=ha, va=va)

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel(r"$k_x (nm^{-1})$")
        ax.set_ylabel(r"$k_y (nm^{-1})$")

        pltutils.set_min_range(abs(max(x) - min(x)) * 1.9, 'x')
        pltutils.set_min_range(abs(max(y) - min(y)) * 1.9, 'y')
        pltutils.despine(trim=True)
        pltutils.add_margin()


def make_lattice(vectors, sublattices, hoppings, min_neighbors=1):
    lat = Lattice(*vectors)
    lat.add_sublattices(*sublattices)
    lat.add_hoppings(*hoppings)
    lat.min_neighbors = min_neighbors
    return lat


def square(a=0.2, t=1):
    lat = Lattice([a, 0], [0, a])
    lat.add_one_sublattice('s', (0, 0))
    lat.add_hoppings(
        [(0,  1), 's', 's', t],
        [(1,  0), 's', 's', t],
        [(1,  1), 's', 's', t],
        [(1, -1), 's', 's', t],
    )
    return lat
