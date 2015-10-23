import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


def test_sweep(baseline, plot):
    @pb.parallelize(v=np.linspace(0, 0.1, 10))
    def factory(v, energy=np.linspace(0, 0.1, 10)):
        model = pb.Model(
            graphene.lattice.monolayer(),
            graphene.shape.hexagon_ac(side_width=20),
            pb.electric.constant(v)
        )

        kpm = pb.greens.make_kpm(model)
        return kpm.deferred_ldos(energy, broadening=0.1, position=[0, 0])

    factory.hooks.status.clear()
    factory.config.pbar_fd = None
    factory.config.filename = None

    labels = dict(title="test sweep", x="V (eV)", y="E (eV)", data="LDOS")
    result = pb.parallel.sweep(factory, labels=labels)

    expected = baseline(result)
    plot(result, expected, 'plot')
    assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)
