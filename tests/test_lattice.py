import pytest

import pybinding as pb
from pybinding.repository import graphene
from pybinding.repository import misc

lattices = {
    'square': pb.lattice.square(a=0.2, t=1),
    'graphene-monolayer': graphene.lattice.monolayer(),
    'graphene-monolayer-alt': graphene.lattice.monolayer_alt(),
    'graphene-monolayer-4atom': graphene.lattice.monolayer_4atom(),
    'graphene-monolayer-nn': graphene.lattice.monolayer_nn(),
    'graphene-bilayer': graphene.lattice.bilayer(),
    'mos2': misc.lattice.mos2(),
}


@pytest.fixture(scope='module', ids=list(lattices.keys()), params=lattices.values())
def lattice(request):
    return request.param


def test_pickle_round_trip(lattice, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    lattice.save(file_name)
    from_file = pb.Lattice.from_file(file_name)

    assert pytest.fuzzy_equal(lattice, from_file)


def test_expected(lattice, baseline, plot):
    expected = baseline(lattice)
    plot(lattice, expected, 'plot')
    assert pytest.fuzzy_equal(lattice, expected)


def test_make_lattice():
    a, t = 1, 1

    lat1 = pb.Lattice([a, 0], [0, a])
    lat1.add_one_sublattice('s', (0, 0))
    lat1.add_hoppings([(0,  1), 's', 's', t],
                      [(1,  0), 's', 's', t])
    lat1.min_neighbors = 2

    lat2 = pb.make_lattice(
        vectors=[[a, 0], [0, a]],
        sublattices=[['s', (0, 0)]],
        hoppings=[[(0, 1), 's', 's', t],
                  [(1, 0), 's', 's', t]],
        min_neighbors=2
    )

    assert pytest.fuzzy_equal(lat1, lat2)
