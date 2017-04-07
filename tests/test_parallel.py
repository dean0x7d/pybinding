import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


def silence_parallel_output(factory):
    factory.hooks.status.clear()
    factory.config.pbar_fd = None
    factory.config.filename = None


def test_sweep(baseline, plot_if_fails):
    @pb.parallelize(v=np.linspace(0, 0.1, 10))
    def factory(v, energy=np.linspace(0, 0.1, 10)):
        model = pb.Model(
            graphene.monolayer(),
            graphene.hexagon_ac(side_width=15),
            pb.constant_potential(v)
        )

        kpm = pb.kpm(model, kernel=pb.lorentz_kernel())
        return kpm.deferred_ldos(energy, broadening=0.15, position=[0, 0], sublattice="B")

    silence_parallel_output(factory)
    labels = dict(title="test sweep", x="V (eV)", y="E (eV)", data="LDOS")
    result = pb.parallel.sweep(factory, labels=labels)

    expected = baseline(result)
    plot_if_fails(result, expected, 'plot')
    assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)


def test_ndsweep(baseline):
    @pb.parallelize(v1=np.linspace(0, 0.1, 5), v2=np.linspace(-0.2, 0.2, 4))
    def factory(v1, v2, energy=np.linspace(0, 0.1, 10)):
        model = pb.Model(
            graphene.monolayer(),
            graphene.hexagon_ac(side_width=15),
            pb.constant_potential(v1),
            pb.constant_potential(v2)
        )

        kpm = pb.kpm(model, kernel=pb.lorentz_kernel())
        return kpm.deferred_ldos(energy, broadening=0.15, position=[0, 0])

    silence_parallel_output(factory)
    result = pb.parallel.ndsweep(factory)

    expected = baseline(result)
    assert pytest.fuzzy_equal(result, expected, rtol=1e-3, atol=1e-6)
