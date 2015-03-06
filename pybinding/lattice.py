import _pybinding


class Lattice(_pybinding.Lattice):
    def __init__(self, min_neighbours=1):
        super().__init__(min_neighbours)
        self.sub = dict()
        self.names = []

    def set_vectors(self, *vectors):
        for vector in vectors:
            self.add_vector(tuple(vector))

    def create_sublattice(self, offset, onsite_potential=0.0, alias=-1, name=""):
        sublattice_id = super().create_sublattice(offset, onsite_potential, alias)
        self.sub[name] = sublattice_id
        self.names.append(name)
        return sublattice_id

    def set_hoppings(self, *hoppings):
        for hop in hoppings:
            self.add_hopping(*hop)

    def set_hopping_matrix(self, *pairs):
        import numpy as np
        for relative_index, matrix in pairs:
            for (from_sub, to_sub), hopping_energy in np.ndenumerate(matrix):
                # skip zero energy hoppings
                if hopping_energy == 0:
                    continue
                # only consider lower triangle values of the (0, 0) matrix
                # the upper triangle implied via Hermitian conjugate
                if all(v == 0 for v in relative_index) and from_sub >= to_sub:
                    continue

                self.add_hopping(relative_index, from_sub, to_sub, hopping_energy)

    def plot(self, **kwargs):
        import pybinding as pb
        import matplotlib.pyplot as plt
        from pybinding.plot.annotate import annotate_box
        ax = plt.gca()
        points = []  # for plot limit detection

        # plot the primitive cell and it's neighbors (using a model... kind of meta)
        model = pb.Model(self, pb.symmetry.translational())
        model.system.plot(boundary_color='black', **kwargs)

        # plot the lattice vectors
        for i, vector in enumerate(self.vectors):
            points += [vector, -vector]
            ax.arrow(0, 0, *vector[:2], color='black', alpha=0.8,
                     head_width=0.02, head_length=0.05, length_includes_head=True)
            annotate_box(r"$v_{}$".format(i+1), xy=vector[:2] / 2, fontcolor='white')

        # annotate the sublattices and neighboring cells
        for sublattice in self.sublattices:
            annotate_box(self.names[sublattice.alias], xy=sublattice.offset[:2])
            for hop in sublattice.hoppings:
                if tuple(hop.relative_index[:2]) == (0, 0):
                    continue  # skip the original cell

                offset = sum(r * v for r, v in zip(hop.relative_index, self.vectors))
                for vector in self.vectors:
                    points += [vector + offset, -vector + offset]
                annotate_box("{}, {}".format(*hop.relative_index[:2]), xy=offset[:2])

        x, y, _ = zip(*points)
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))


def square(a=0.2, t=1):
    lat = Lattice()
    lat.set_vectors([a, 0], [0, a])

    s = lat.create_sublattice((0, 0))
    lat.set_hoppings(
        [(0,  1), s, s, t],
        [(1,  0), s, s, t],
        [(1,  1), s, s, t],
        [(1, -1), s, s, t],
    )
    return lat
