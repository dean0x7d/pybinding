import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene
from pybinding.support.deprecated import LoudDeprecationWarning

lattices = {
    "graphene-monolayer": graphene.monolayer(),
    "graphene-monolayer-nn": graphene.monolayer(2),
    "graphene-monolayer-4atom": graphene.monolayer_4atom(),
    "graphene-bilayer": graphene.bilayer(),
}


@pytest.fixture(scope='module', ids=list(lattices.keys()), params=lattices.values())
def lattice(request):
    return request.param


def test_pickle_round_trip(lattice):
    import pickle
    unpickled = pickle.loads(pickle.dumps(lattice))
    assert pytest.fuzzy_equal(lattice, unpickled)


def test_expected(lattice, baseline, plot_if_fails):
    expected = baseline(lattice)
    plot_if_fails(lattice, expected, "plot")
    assert pytest.fuzzy_equal(lattice, expected)


def test_init():
    lat1d = pb.Lattice(1)
    assert lat1d.ndim == 1
    assert len(lat1d.vectors) == 1
    assert pytest.fuzzy_equal(lat1d.vectors[0], [1, 0, 0])

    lat2d = pb.Lattice([1, 0], [0, 1])
    assert lat2d.ndim == 2
    assert len(lat2d.vectors) == 2
    assert pytest.fuzzy_equal(lat2d.vectors[0], [1, 0, 0])
    assert pytest.fuzzy_equal(lat2d.vectors[1], [0, 1, 0])

    lat3d = pb.Lattice([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert lat3d.ndim == 3
    assert len(lat3d.vectors) == 3
    assert pytest.fuzzy_equal(lat3d.vectors[0], [1, 0, 0])
    assert pytest.fuzzy_equal(lat3d.vectors[1], [0, 1, 0])
    assert pytest.fuzzy_equal(lat3d.vectors[2], [0, 0, 1])


def test_add_sublattice(capsys):
    lat = pb.Lattice(1)
    assert lat.nsub == 0

    lat.add_one_sublattice("A", 0.0)
    assert lat.nsub == 1

    lat.add_sublattices(("B", 0.1),
                        ("C", 0.2))
    assert lat.nsub == 3

    subs = lat.sublattices
    assert len(subs) == 3
    assert all(v in subs for v in ("A", "B", "C"))

    assert pytest.fuzzy_equal(subs["A"].position, [0, 0, 0])
    assert subs["A"].energy == 0
    assert subs["A"].unique_id == 0
    assert subs["A"].alias_id == 0

    assert pytest.fuzzy_equal(subs["B"].position, [0.1, 0, 0])
    assert subs["B"].energy == 0
    assert subs["B"].unique_id == 1
    assert subs["B"].alias_id == 1

    assert pytest.fuzzy_equal(subs["C"].position, [0.2, 0, 0])
    assert subs["C"].energy == 0
    assert subs["C"].unique_id == 2
    assert subs["C"].alias_id == 2

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("", 0)
    assert "Sublattice name can't be blank" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("A", 0)
    assert "Sublattice 'A' already exists" in str(excinfo.value)

    with pytest.warns(LoudDeprecationWarning):
        assert lat["A"] == "A"
    capsys.readouterr()


def test_add_multiorbital_sublattice():
    lat = pb.Lattice([1, 0], [0, 1])
    lat.add_one_sublattice("A", [0, 0])
    assert lat.nsub == 1

    lat.add_one_sublattice("B", [0, 0], [1, 2, 3])
    assert pytest.fuzzy_equal(lat.sublattices["B"].energy, [[1, 0, 0],
                                                            [0, 2, 0],
                                                            [0, 0, 3]])

    lat.add_one_sublattice("C", [0, 0], [[1, 2, 3],
                                         [0, 4, 5],
                                         [0, 0, 6]])
    assert pytest.fuzzy_equal(lat.sublattices["C"].energy, [[1, 2, 3],
                                                            [2, 4, 5],
                                                            [3, 5, 6]])

    lat.add_one_sublattice("D", [0, 0], [[1, 2j,  3],
                                         [0,  4, 5j],
                                         [0,  0,  6]])
    assert pytest.fuzzy_equal(lat.sublattices["D"].energy, [[  1, 2j,  3],
                                                            [-2j,  4, 5j],
                                                            [  3, -5j, 6]])

    lat.add_one_sublattice("E", [0, 0], [[1, 2, 3],
                                         [2, 4, 5],
                                         [3, 5, 6]])
    assert pytest.fuzzy_equal(lat.sublattices["E"].energy, [[1, 2, 3],
                                                            [2, 4, 5],
                                                            [3, 5, 6]])
    assert lat.nsub == 5

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("zero-dimensional", [0, 0], [])
    assert "can't be zero-dimensional" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("complex onsite energy", [0, 0], [1j, 2j, 3j])
    assert "must be a real vector or a square matrix" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("not square", [0, 0], [[1, 2, 3],
                                                      [4, 5, 6]])
    assert "must be a real vector or a square matrix" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("not square", [0, 0], [[1j,  2,  3],
                                                      [ 2, 4j,  5],
                                                      [ 3,  5, 6j]])
    assert "The main diagonal of the onsite hopping term must be real" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_sublattice("not Hermitian", [0, 0], [[1, 2, 3],
                                                         [4, 5, 6],
                                                         [7, 8, 9]])
    assert "The onsite hopping matrix must be upper triangular or Hermitian" in str(excinfo.value)


def test_add_sublattice_alias(capsys):
    lat = pb.Lattice([1, 0], [0, 1])
    lat.add_sublattices(("A", [0.0, 0.5]),
                        ("B", [0.5, 0.0]))

    c_position = [0, 9]
    lat.add_one_alias("C", "A", c_position)
    assert lat.sublattices["C"].unique_id != lat.sublattices["A"].unique_id
    assert lat.sublattices["C"].alias_id == lat.sublattices["A"].alias_id

    model = pb.Model(lat)
    c_index = model.system.find_nearest(c_position)
    assert model.system.sublattices[c_index] == lat.sublattices["C"].alias_id
    assert c_index in np.argwhere(model.system.sublattices == "A")

    with pytest.raises(IndexError) as excinfo:
        lat.add_one_alias("D", "bad_name", [0, 0])
    assert "There is no sublattice named 'bad_name'" in str(excinfo.value)

    with pytest.warns(LoudDeprecationWarning):
        lat.add_one_sublattice("Z", c_position, alias="A")
    capsys.readouterr()


def test_add_hopping(capsys):
    lat = pb.Lattice([1, 0], [0, 1])
    lat.add_sublattices(("A", [0.0, 0.5]),
                        ("B", [0.5, 0.0]))
    lat.add_hoppings(([0,  0], "A", "B", 1),
                     ([1, -1], "A", "B", 1),
                     ([0, -1], "A", "B", 2))

    assert lat.nhop == 2
    assert lat.hoppings["__anonymous__0"].family_id == 0
    assert lat.hoppings["__anonymous__0"].energy == 1
    assert lat.hoppings["__anonymous__1"].family_id == 1
    assert lat.hoppings["__anonymous__1"].energy == 2

    lat.add_hoppings(([0,  1], "A", "B", 1))
    assert lat.nhop == 2

    lat.add_hoppings(([1,  0], "A", "B", 3))
    assert lat.nhop == 3

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_hopping([0, 0], "A", "B", 1)
    assert "hopping already exists" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_hopping([0, 0], "A", "A", 1)
    assert "Don't define onsite energy here" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        lat.add_one_hopping([0, 0], "C", "A", 1)
    assert "There is no sublattice named 'C'" in str(excinfo.value)

    lat.register_hopping_energies({
        "t_nn": 0.1,
        "t_nnn": 0.01
    })

    assert lat.nhop == 5
    assert lat.hoppings["t_nn"].energy == 0.1
    assert lat.hoppings["t_nnn"].energy == 0.01

    lat.add_one_hopping([0, 1], "A", "A", "t_nn")

    with pytest.raises(RuntimeError) as excinfo:
        lat.register_hopping_energies({"": 0.0})
    assert "Hopping name can't be blank" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.register_hopping_energies({"t_nn": 0.2})
    assert "Hopping 't_nn' already exists" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        lat.add_one_hopping((0, 1), "A", "A", "tt")
    assert "There is no hopping named 'tt'" in str(excinfo.value)

    with pytest.warns(LoudDeprecationWarning):
        assert lat("t_nn") == "t_nn"
    capsys.readouterr()


def test_add_matrix_hopping():
    lat = pb.Lattice([1, 0], [0, 1])
    lat.add_sublattices(("A", [0.0, 0.5]),
                        ("B", [0.5, 0.0]))
    lat.add_hoppings(([0, 0], "A", "B", 1),
                     ([1, -1], "A", "B", 1),
                     ([0, -1], "A", "B", 2))
    assert lat.nsub == 2
    assert lat.nhop == 2

    lat.add_sublattices(("A2", [0, 0], [1, 2]),
                        ("B2", [0, 0], [1, 2]),
                        ("C3", [0, 0], [1, 2, 3]))
    assert lat.nsub == 5

    lat.register_hopping_energies({"t22": [[1, 2],
                                           [3, 4]],
                                   "t23": [[1, 2, 3],
                                           [4, 5, 6]]})
    assert lat.nhop == 4

    with pytest.raises(RuntimeError) as excinfo:
        lat.register_hopping_energies({"zero-dimensional": []})
    assert "can't be zero-dimensional" in str(excinfo.value)

    lat.add_hoppings(([0, 0], "A2", "B2", "t22"),
                     ([1, 0], "A2", "A2", "t22"),
                     ([0, 0], "A2", "C3", "t23"),
                     ([1, 0], "A2", "C3", "t23"))

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_hopping([0, 0], 'A2', 'A2', "t22")
    assert "Don't define onsite energy here" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_hopping([0, 0], 'B2', 'C3', "t22")
    assert "mismatch: from 'B2' (2) to 'C3' (3) with matrix 't22' (2, 2)" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        lat.add_one_hopping([0, 0], 'C3', 'B2', "t23")
    assert "mismatch: from 'C3' (3) to 'B2' (2) with matrix 't23' (2, 3)" in str(excinfo.value)


def test_builder():
    """Builder pattern methods"""
    lattice = pb.Lattice([1, 0], [0, 1])

    copy = lattice.with_offset([0, 0.5])
    assert pytest.fuzzy_equal(copy.offset, [0, 0.5, 0])
    assert pytest.fuzzy_equal(lattice.offset, [0, 0, 0])

    copy = lattice.with_min_neighbors(5)
    assert copy.min_neighbors == 5
    assert lattice.min_neighbors == 1


def test_brillouin_zone():
    from math import pi, sqrt

    lat = pb.Lattice(a1=1)
    assert pytest.fuzzy_equal(lat.brillouin_zone(), [-pi, pi])

    lat = pb.Lattice(a1=[0, 1], a2=[0.5, 0.5])
    assert pytest.fuzzy_equal(lat.brillouin_zone(),
                              [[0, -2 * pi], [2 * pi, 0], [0, 2 * pi], [-2 * pi, 0]])

    # Identical lattices represented using acute and obtuse angles between primitive vectors
    acute = pb.Lattice(a1=[1, 0], a2=[1/2, 1/2 * sqrt(3)])
    obtuse = pb.Lattice(a1=[1/2, 1/2 * sqrt(3)], a2=[1/2, -1/2 * sqrt(3)])
    assert pytest.fuzzy_equal(acute.brillouin_zone(), obtuse.brillouin_zone())
