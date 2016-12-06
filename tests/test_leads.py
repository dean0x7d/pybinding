import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene, examples


@pytest.fixture
def ring_model():
    def ring(inner_radius, outer_radius):
        def contains(x, y, _):
            r = np.sqrt(x**2 + y**2)
            return np.logical_and(inner_radius < r, r < outer_radius)
        return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])

    return pb.Model(graphene.monolayer(), ring(3, 5))


@pytest.fixture
def square_model(width=2, height=3):
    def square_lattice(d=1, t=1):
        lat = pb.Lattice(a1=[d, 0], a2=[0, d])
        lat.add_sublattices(('A', [0, 0]))
        lat.add_hoppings(
            ([0, 1], 'A', 'A', -t),
            ([1, 0], 'A', 'A', -t),
        )
        return lat
    return pb.Model(square_lattice(), pb.rectangle(width, height))


def linear_onsite(k=1):
    @pb.onsite_energy_modifier
    def onsite(x):
        return k * x
    return onsite


def linear_hopping(k=1):
    @pb.hopping_energy_modifier
    def hopping(x1, x2):
        return 0.5 * k * (x1 + x2)
    return hopping


def test_api(ring_model):
    with pytest.raises(RuntimeError) as excinfo:
        ring_model.attach_lead(0, pb.line(0, 0))
    assert "Lead direction must be one of" in str(excinfo.value)

    for direction in [3, -3]:
        with pytest.raises(RuntimeError) as excinfo:
            ring_model.attach_lead(direction, pb.line(0, 0))
        assert "not valid for a 2D lattice" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.Model(examples.chain_lattice()).attach_lead(2, pb.line(0, 0))
    assert "Attaching leads to 1D lattices is not supported" in str(excinfo.value)


def test_partial_miss(ring_model):
    with pytest.raises(RuntimeError) as excinfo:
        ring_model.attach_lead(2, pb.line([0, -5], [5, -5]))
        assert ring_model.system
    assert "partially misses main structure" in str(excinfo.value)


def test_complete_miss(ring_model):
    with pytest.raises(RuntimeError) as excinfo:
        ring_model.attach_lead(2, pb.line([4, -5], [5, -5]))
        assert ring_model.system
    assert "completely misses main structure" in str(excinfo.value)


def test_empty(ring_model):
    with pytest.raises(RuntimeError) as excinfo:
        ring_model.attach_lead(2, pb.line(0, 0))
        assert ring_model.system
    assert "no sites in lead junction" in str(excinfo.value)


def test_attach():
    """Attach 2 leads to a square lattice system"""
    w, h = 2, 3
    model = square_model(w, h)
    model.attach_lead(-1, pb.line([0, -h/2], [0, h/2]))
    model.attach_lead(+1, pb.line([0, -h/2], [0, h/2]))
    assert len(model.leads) == 2
    assert np.all(model.leads[0].indices == [0, 2, 4])
    assert np.all(model.leads[1].indices == [1, 3, 5])

    # Linear hopping grows from lead 0 to system to lead 1
    model.add(linear_hopping())
    assert model.leads[0].h1.data.max() < model.hamiltonian.data.min()
    assert model.hamiltonian.data.max() < model.leads[1].h1.data.min()

    # With the linear hopping modifier, the h1 hoppings should be equal to the
    # x position between the lead and main system
    x_mid_0 = (model.system.x.min() + model.leads[0].system.x.max()) / 2
    assert np.allclose(model.leads[0].h1.data, x_mid_0)
    x_mid_1 = (model.system.x.max() + model.leads[1].system.x.min()) / 2
    assert np.allclose(model.leads[1].h1.data, x_mid_1)

    # Linear onsite potential grows from lead 0 to system to lead 1
    model.add(linear_onsite())
    assert model.leads[0].h0.diagonal().max() < model.hamiltonian.diagonal().min()
    assert model.hamiltonian.diagonal().max() < model.leads[1].h0.diagonal().min()
