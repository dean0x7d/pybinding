import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


def test_sweep(baseline, plot):
    def produce(var):
        model = pb.Model(
            graphene.lattice.monolayer(),
            graphene.shape.hexagon_ac(side_width=20),
            pb.electric.constant(var)
        )

        kpm = pb.greens.make_kpm(model)
        return kpm.deferred_ldos(np.linspace(0, 0.1, 10), broadening=0.1, position=[0, 0])

    result = pb.parallel.sweep(
        variables=np.linspace(0, 0.1, 10),
        produce=produce,
        pbar_fd=None, silent=True,
        labels=dict(title="test sweep", x="V (eV)", y="E (eV)", data="LDOS")
    )

    expected = baseline(result)
    plot(result, expected, 'plot')
    assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)
