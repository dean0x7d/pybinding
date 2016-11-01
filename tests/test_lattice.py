import pytest

import pybinding as pb
from pybinding.repository import graphene

lattices = {
    'graphene-monolayer': graphene.monolayer(),
    'graphene-monolayer-alt': graphene.monolayer_alt(),
    'graphene-monolayer-4atom': graphene.monolayer_4atom(),
    'graphene-monolayer-nn': graphene.monolayer(2),
    'graphene-bilayer': graphene.bilayer(),
}


@pytest.fixture(scope='module', ids=list(lattices.keys()), params=lattices.values())
def lattice(request):
    return request.param


@pytest.fixture
def mock_lattice():
    a_cc, a, t = 1, 1.73, 1
    lat = pb.Lattice([a, 0], [0.5 * a, 0.866 * a])
    lat.add_sublattices(
        ['a', (0, -a_cc/2)],
        ['b', (0,  a_cc/2)]
    )
    lat.add_hoppings(
        [(0,  0), 'a', 'b', t],
        [(1, -1), 'a', 'b', t],
        [(0, -1), 'a', 'b', t]
    )
    lat.min_neighbors = 2
    return lat


def test_add_sublattice(mock_lattice):
    with pytest.raises(RuntimeError) as excinfo:
        mock_lattice.add_one_sublattice('a', (0, 0))
    assert "Sublattice 'a' already exists" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        assert mock_lattice['c']
    assert "There is no sublattice named 'c'" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        assert mock_lattice[5]
    assert "There is no sublattice with ID = 5" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        for i in range(127):
            mock_lattice.add_one_sublattice(str(i), (0, 0))
    assert "Cannot create more sublattices" in str(excinfo.value)


def test_add_sublattice_alias(mock_lattice):
    c_position = [0, 9]
    mock_lattice.add_one_sublattice('c', c_position, alias='a')
    model = pb.Model(mock_lattice)
    c_index = model.system.find_nearest(c_position)

    assert mock_lattice['c'] != mock_lattice['a']
    assert model.system.sublattices[c_index] == mock_lattice['a']

    with pytest.raises(KeyError) as excinfo:
        mock_lattice.add_one_sublattice('d', [0, 0], alias='bad_name')
    assert "There is no sublattice named 'bad_name'" in str(excinfo.value)


def test_add_hopping(mock_lattice):
    with pytest.raises(RuntimeError) as excinfo:
        mock_lattice.add_one_hopping((0,  0), 'a', 'b', 1)
    assert "hopping already exists" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        mock_lattice.add_one_hopping((0, 0), 'a', 'a', 1)
    assert "Don't define onsite energy here" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        mock_lattice.add_one_hopping((0, 0), 'c', 'a', 1)
    assert "There is no sublattice named 'c'" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        assert mock_lattice('t_nn')
    assert "There is no hopping named 't_nn'" in str(excinfo.value)

    mock_lattice.register_hopping_energies({
        't_nn': 0.1,
        't_nnn': 0.01
    })
    assert mock_lattice('t_nn') == 1
    assert mock_lattice('t_nnn') == 2

    mock_lattice.add_one_hopping((0, 1), 'a', 'a', 't_nn')

    with pytest.raises(RuntimeError) as excinfo:
        mock_lattice.register_hopping_energies({'t_nn': 0.2})
    assert "Hopping 't_nn' already exists" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        mock_lattice.add_one_hopping((0, 1), 'a', 'a', 'tt')
    assert "There is no hopping named 'tt'" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        for i in range(1, 128):
            mock_lattice.add_one_hopping((0, i), 'a', 'b', i)
    assert "Can't create any more hoppings" in str(excinfo.value)


def test_builder():
    """Builder pattern methods"""
    lattice = pb.Lattice([1, 0], [0, 1])

    copy = lattice.with_offset([0, 0.5])
    assert pytest.fuzzy_equal(copy.offset, [0, 0.5, 0])
    assert pytest.fuzzy_equal(lattice.offset, [0, 0, 0])

    copy = lattice.with_min_neighbors(5)
    assert copy.min_neighbors == 5
    assert lattice.min_neighbors == 1


def test_pickle_round_trip(lattice, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    pb.save(lattice, file_name)
    from_file = pb.load(file_name)

    assert lattice.sub_name_map == from_file.sub_name_map
    assert lattice.hop_name_map == from_file.hop_name_map
    assert pytest.fuzzy_equal(lattice, from_file)


def test_expected(lattice, baseline, plot_if_fails):
    expected = baseline(lattice)
    plot_if_fails(lattice, expected, 'plot')

    assert lattice.sub_name_map == expected.sub_name_map
    assert lattice.hop_name_map == expected.hop_name_map
    assert pytest.fuzzy_equal(lattice, expected)


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
