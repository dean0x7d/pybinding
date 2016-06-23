import pytest

import pybinding as pb
from pybinding.repository import graphene

models = {
    'graphene-monolayer': [graphene.monolayer(), graphene.hexagon_ac(1)],
    'graphene-monolayer-alt': [graphene.monolayer_alt(), pb.rectangle(1.6, 1.4)],
    'graphene-monolayer-4atom': [graphene.monolayer_4atom()],
    'graphene-monolayer-nn': [graphene.monolayer(2), pb.regular_polygon(6, 0.9)],
    'graphene-monolayer-periodic-1d': [graphene.monolayer(), pb.primitive(5, 5),
                                       pb.translational_symmetry(a1=True, a2=False)],
    'graphene-monolayer-periodic-1d-alt': [graphene.monolayer_4atom(), pb.rectangle(1),
                                           pb.translational_symmetry(a1=False, a2=0.6)],
    'graphene-monolayer-periodic-2d': [graphene.monolayer(), pb.primitive(a1=5, a2=5),
                                       pb.translational_symmetry(a1=1, a2=1)],
    'graphene-monolayer-4atom-periodic-2d': [graphene.monolayer_4atom(), pb.rectangle(1),
                                             pb.translational_symmetry(a1=0.6, a2=0.6)],
    'graphene-bilayer': [graphene.bilayer(), graphene.hexagon_ac(0.6)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model(request):
    return pb.Model(*request.param)


def test_api():
    model = pb.Model(graphene.monolayer(), pb.primitive(2, 2))
    system = model.system

    idx = system.num_sites // 2
    assert idx == system.find_nearest(system.xyz[idx])
    assert idx == system.find_nearest(system.xyz[idx], system.sublattices[idx])
    assert system.find_nearest([0, 0], 'A') != system.find_nearest([0, 0], 'B')

    invalid_sublattice = 99
    with pytest.raises(KeyError) as excinfo:
        system.find_nearest([0, 0], invalid_sublattice)
    assert "There is no sublattice" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        system.find_nearest([0, 0], 'invalid_sublattice')
    assert "There is no sublattice" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        system.hoppings.data += 1
    assert "read-only" in str(excinfo.value)


def test_sites():
    model = pb.Model(graphene.monolayer(), pb.primitive(2, 2))
    system = model.system

    sites = pb.system.Sites(system.positions, system.sublattices)
    idx = system.num_sites // 2
    assert idx == sites.find_nearest(system.xyz[idx])
    assert idx == sites.find_nearest(system.xyz[idx], system.sublattices[idx])
    assert sites.find_nearest([0, 0], 'A') != sites.find_nearest([0, 0], 'B')


def test_pickle_round_trip(model, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    pb.save(model.system, file_name)
    from_file = pb.load(file_name)

    assert pytest.fuzzy_equal(model.system, from_file)


def test_expected(model, baseline, plot_if_fails):
    system = model.system
    expected = baseline(system)
    plot_if_fails(system, expected, 'plot')
    assert pytest.fuzzy_equal(system, expected, 1.e-4, 1.e-6)


def test_system_plot(compare_figure):
    model = pb.Model(graphene.bilayer(), graphene.hexagon_ac(0.1))
    with compare_figure() as chk:
        model.system.plot()
    assert chk.passed
