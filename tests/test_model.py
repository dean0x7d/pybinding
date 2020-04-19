import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene
from scipy.sparse import csr_matrix


def point_to_same_memory(a, b):
    """Check if two numpy arrays point to the same data in memory"""
    return a.data == b.data


@pytest.fixture(scope='module')
def model():
    return pb.Model(graphene.monolayer())


def test_api():
    lattice = graphene.monolayer()
    shape = pb.rectangle(1)
    model = pb.Model(lattice, shape)

    assert model.lattice is lattice
    assert model.shape is shape

    # empty sequences are no-ops
    model.add(())
    model.add([])

    with pytest.raises(RuntimeError) as excinfo:
        model.add(None)
    assert "None" in str(excinfo.value)


def test_report(model):
    report = model.report()
    assert "2 lattice sites" in report
    assert "2 non-zero values" in report


def test_hamiltonian(model):
    """Must be in the correct format and point to memory allocated in C++ (no copies)"""
    h = model.hamiltonian
    assert isinstance(h, csr_matrix)
    assert h.dtype == np.float32
    assert h.shape == (2, 2)
    assert pytest.fuzzy_equal(h.data, [graphene.t] * 2)
    assert pytest.fuzzy_equal(h.indices, [1, 0])
    assert pytest.fuzzy_equal(h.indptr, [0, 1, 2])

    assert h.data.flags['OWNDATA'] is False
    assert h.data.flags['WRITEABLE'] is False

    with pytest.raises(ValueError) as excinfo:
        h.data += 1
    assert "read-only" in str(excinfo.value)

    h2 = model.hamiltonian
    assert h2.data is not h.data
    assert point_to_same_memory(h2.data, h.data)


def test_multiorbital_hamiltonian():
    """For multi-orbital lattices the Hamiltonian size is larger than the number of sites"""
    def lattice():
        lat = pb.Lattice([1])
        lat.add_sublattices(("A", [0], [[1, 3j],
                                        [0, 2]]))
        lat.register_hopping_energies({
            "t22": [[0, 1],
                    [2, 3]],
            "t11": 1,  # incompatible hopping - it's never used so it shouldn't raise any errors
        })
        lat.add_hoppings(([1], "A", "A", "t22"))
        return lat

    model = pb.Model(lattice(), pb.primitive(3))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[  1, 3j],
                                          [-3j, 2]])
    assert pytest.fuzzy_equal(h[:2, 2:4], [[0, 1],
                                           [2, 3]])

    @pb.onsite_energy_modifier
    def onsite(energy, x, sub_id):
        return 3 * energy + sub_id.eye * 0 * x

    @pb.hopping_energy_modifier
    def hopping(energy):
        return 2 * energy

    model = pb.Model(lattice(), pb.primitive(3), onsite, hopping)
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[  3, 9j],
                                          [-9j,  6]])
    assert pytest.fuzzy_equal(h[:2, 2:4], [[0, 2],
                                           [4, 6]])
    assert pytest.fuzzy_equal(h[2:4, 4:6], [[0, 2],
                                            [4, 6]])

    def lattice_with_zero_diagonal():
        lat = pb.Lattice([1])
        lat.add_sublattices(("A", [0], [[0, 3j],
                                        [0,  0]]))
        return lat

    model = pb.Model(lattice_with_zero_diagonal(), pb.primitive(3))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[0, 3j],
                                          [-3j, 0]])


def test_complex_multiorbital_hamiltonian():
    def checkerboard_lattice(delta, t):
        lat = pb.Lattice(a1=[1, 0], a2=[0, 1])
        lat.add_sublattices(('A', [0, 0],    -delta),
                            ('B', [1/2, 1/2], delta))
        lat.add_hoppings(
            ([0,   0], 'A', 'B', t),
            ([0,  -1], 'A', 'B', t),
            ([-1,  0], 'A', 'B', t),
            ([-1, -1], 'A', 'B', t),
        )
        return lat

    hopp_t = np.array([[2 + 2j, 3 + 3j],
                       [4 + 4j, 5 + 5j]])      # multi-orbital hopping
    onsite_en = np.array([[1, 1j], [-1j, 1]])  # onsite energy

    model = pb.Model(checkerboard_lattice(onsite_en, hopp_t),
                     pb.translational_symmetry(True, True))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 2
    assert h.shape[0] == 4
    assert pytest.fuzzy_equal(h, h.T.conjugate())  # check if Hermitian
    assert pytest.fuzzy_equal(h[:2, :2], -h[-2:, -2:])  # onsite energy on A and B is opposite
    assert pytest.fuzzy_equal(h[:2, 2:4], 4 * hopp_t)  # hopping A <-> B is 4 * hopp_t
