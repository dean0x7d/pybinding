import pytest
import numpy as np

import pybinding as pb
from pybinding.repository import graphene

models = {
    'graphene-pristine': [graphene.lattice.monolayer(), pb.shape.rectangle(70)],
    'graphene-magnetic_field': [graphene.lattice.monolayer(), pb.shape.rectangle(70),
                                pb.magnetic.constant(60)],
}


@pytest.fixture(scope='module', ids=list(models.keys()), params=models.values())
def kpm(request):
    model = pb.Model(*request.param)
    return pb.greens.make_kpm(model)


def test_ldos(kpm, baseline, plot):
    result = kpm.calc_ldos(energy=np.linspace(-0.6, 0.6, 200), broadening=0.03, position=(0, 0))
    expected = baseline(result)

    plot(result, expected, 'plot')
    assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)
