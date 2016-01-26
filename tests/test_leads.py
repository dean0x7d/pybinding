import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


@pytest.fixture
def model():
    def ring(inner_radius, outer_radius):
        def contains(x, y, _):
            r = np.sqrt(x**2 + y**2)
            return np.logical_and(inner_radius < r, r < outer_radius)
        return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])

    return pb.Model(graphene.monolayer(), ring(3, 5))


def test_api(model):
    with pytest.raises(RuntimeError) as excinfo:
        model.attach_lead(1)
    assert "Bad arguments" in str(excinfo.value)

    for direction in [0, 4, -4]:
        with pytest.raises(RuntimeError) as excinfo:
            model.attach_lead(direction, [0, 0], [0, 0])
        assert "Lead direction must be one of" in str(excinfo.value)


def test_partial_miss(model):
    with pytest.raises(RuntimeError) as excinfo:
        model.attach_lead(2, [0, -5], [5, -5])
        assert model.system
    assert "partially misses main structure" in str(excinfo.value)


def test_complete_miss(model):
    with pytest.raises(RuntimeError) as excinfo:
        model.attach_lead(2, [4, -5], [5, -5])
        assert model.system
    assert "completely misses main structure" in str(excinfo.value)


def test_empty(model):
    with pytest.raises(RuntimeError) as excinfo:
        model.attach_lead(2, [0, 0], [0, 0])
        assert model.system
    assert "no sites in lead junction" in str(excinfo.value)


def test_expected(model, baseline):
    model.attach_lead(1, [0, -1], [0, 1])
    model.attach_lead(-1, [0, -1], [0, 1])
    model.attach_lead(2, [-1, -5], [1, -5])
    model.attach_lead(-2, [-1, 5], [1, 5])

    ports = model.system.ports
    expected = baseline(ports)
    assert pytest.fuzzy_equal(ports, expected, 1.e-4, 1.e-6)
