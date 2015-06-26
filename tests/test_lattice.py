import pytest

import pybinding as pb
from pybinding.repository import graphene

lattices = [
    pb.lattice.square(a=0.2, t=1),
    graphene.lattice.monolayer(),
    graphene.lattice.monolayer_alt(),
    graphene.lattice.monolayer_4atom(),
    graphene.lattice.monolayer_nn(),
    graphene.lattice.bilayer(),
]


@pytest.fixture(scope='module', params=lattices)
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
