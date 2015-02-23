import _pybinding


class Lattice(_pybinding.Lattice):
    def __init__(self, min_neighbours = 1):
        super().__init__(min_neighbours)
        self.sub = dict()

    def set_vectors(self, *vectors):
        for vector in vectors:
            self.add_vector(tuple(vector))

    def create_sublattice(self, offset, onsite_potential=0.0, alias=-1, name=""):
        sublattice_id = super().create_sublattice(offset, onsite_potential, alias)
        self.sub[name] = sublattice_id
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
