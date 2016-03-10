import pytest
import numpy as np

import pybinding as pb
from pybinding.repository import graphene

models = {
    'graphene-pristine': [graphene.monolayer(), pb.rectangle(30)],
    'graphene-pristine-oversized': [graphene.monolayer(), pb.rectangle(45)],
    'graphene-const_potential': [graphene.monolayer(), pb.rectangle(30),
                                 pb.constant_potential(0.5)],
    'graphene-magnetic_field': [graphene.monolayer(), pb.rectangle(30),
                                graphene.constant_magnetic_field(800)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def kpm(request):
    model = pb.Model(*request.param)
    return [pb.greens.kpm(model, optimization_level=i) for i in range(3)]


def test_ldos(kpm, baseline, plot_if_fails):
    energy = np.linspace(-2, 2, 100)
    results = [k.calc_ldos(energy, broadening=0.06, position=(0, 0)) for k in kpm]

    expected = pb.results.LDOS(energy, baseline(results[0].ldos))
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    assert pytest.fuzzy_equal(results[0], expected, rtol=1e-3, atol=1e-6)
    assert pytest.fuzzy_equal(results[1], expected, rtol=1e-3, atol=1e-6)
    assert pytest.fuzzy_equal(results[2], expected, rtol=1e-3, atol=1e-6)


def test_kpm_reuse():
    """KPM should return the same result when a single object is used for multiple calculations"""
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10))
    kpm = pb.greens.kpm(model)
    energy = np.linspace(-5, 5, 50)
    broadening = 0.1

    for position in [0, 0], [6, 0]:
        actual = kpm.calc_ldos(energy, broadening, position)
        expected = pb.greens.kpm(model).calc_ldos(energy, broadening, position)
        assert pytest.fuzzy_equal(actual, expected, rtol=1e-3, atol=1e-6)


def test_ldos_sublattice():
    """LDOS for A and B sublattices should be antisymmetric for graphene with a mass term"""
    model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(10), graphene.mass_term(1))
    kpm = pb.greens.kpm(model)

    a, b = (kpm.calc_ldos(np.linspace(-5, 5, 50), 0.1, [0, 0], sub) for sub in ('A', 'B'))
    assert pytest.fuzzy_equal(a.ldos, b.ldos[::-1], rtol=1e-3, atol=1e-6)
