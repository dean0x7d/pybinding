import pytest

import pybinding as pb
from pybinding.repository import graphene, examples

models = {
    'square': [examples.square_lattice(d=0.2, t=1), pb.shape.rectangle(1)],
    'square-periodic-2d': [examples.square_lattice(d=0.2, t=1), pb.shape.rectangle(2),
                           pb.symmetry.translational(v1=0.6, v2=0.6)],
    'graphene-monolayer': [graphene.lattice.monolayer(), graphene.shape.hexagon_ac(1)],
    'graphene-monolayer-alt': [graphene.lattice.monolayer_alt(), pb.shape.rectangle(1.6, 1.4)],
    'graphene-monolayer-4atom': [graphene.lattice.monolayer_4atom()],
    'graphene-monolayer-nn': [graphene.lattice.monolayer_nn(), pb.shape.regular_polygon(6, 0.9)],
    'graphene-monolayer-periodic-1d': [graphene.lattice.monolayer(), pb.shape.primitive(5, 5),
                                       pb.symmetry.translational(v1=0)],
    'graphene-monolayer-periodic-1d-alt': [graphene.lattice.monolayer_4atom(),
                                           pb.shape.rectangle(1),
                                           pb.symmetry.translational(v2=0.6)],
    'graphene-monolayer-periodic-2d': [graphene.lattice.monolayer_4atom(), pb.shape.rectangle(1),
                                       pb.symmetry.translational(v1=0.6, v2=0.6)],
    'graphene-bilayer': [graphene.lattice.bilayer(), graphene.shape.hexagon_ac(0.6)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def model(request):
    return pb.Model(*request.param)


def test_pickle_round_trip(model, tmpdir):
    file_name = str(tmpdir.join('file.npz'))
    model.system.save(file_name)
    from_file = pb.system.System.from_file(file_name)

    assert pytest.fuzzy_equal(model.system, from_file)


def test_system(model, baseline, plot_if_fails):
    system = model.system
    expected = baseline(system)

    plot_if_fails(system, expected, 'plot')
    assert pytest.fuzzy_equal(system, expected, 1.e-4, 1.e-6)

    idx = system.num_sites // 2
    assert idx == system.find_nearest(system.xyz[idx])
    assert idx == system.find_nearest(system.xyz[idx], system.sublattices[idx])
    assert idx == expected.find_nearest(expected.xyz[idx])
    assert idx == expected.find_nearest(expected.xyz[idx], expected.sublattices[idx])


def test_system_plot(compare_figure):
    model = pb.Model(graphene.lattice.bilayer(), graphene.shape.hexagon_ac(0.1))
    with compare_figure() as chk:
        model.system.plot()
    assert chk.passed
