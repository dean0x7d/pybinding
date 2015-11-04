import pytest
import numpy as np

import pybinding as pb
from pybinding.repository import graphene

models = {
    'graphene-pristine': [graphene.lattice.monolayer(), pb.shape.rectangle(30)],
    'graphene-pristine-oversized': [graphene.lattice.monolayer(), pb.shape.rectangle(45)],
    'graphene-const_potential': [graphene.lattice.monolayer(), pb.shape.rectangle(30),
                                 pb.electric.constant(0.5)],
    'graphene-magnetic_field': [graphene.lattice.monolayer(), pb.shape.rectangle(30),
                                pb.magnetic.constant(800)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def kpm(request):
    model = pb.Model(*request.param)
    return [pb.greens.make_kpm(model, optimization_level=i) for i in range(3)]


def test_ldos(kpm, baseline, plot_if_fails):
    results = [k.calc_ldos(energy=np.linspace(-2, 2, 100), broadening=0.06, position=(0, 0))
               for k in kpm]
    expected = baseline(results[0])
    for i in range(len(results)):
        plot_if_fails(results[i], expected, 'plot', label=i)

    assert pytest.fuzzy_equal(results[0], expected, rtol=1e-3, atol=1e-6)
    assert pytest.fuzzy_equal(results[1], expected, rtol=1e-3, atol=1e-6)
    assert pytest.fuzzy_equal(results[2], expected, rtol=1e-3, atol=1e-6)
