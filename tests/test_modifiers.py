import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


one, zero = np.ones(1), np.zeros(1)
complex_one = np.ones(1, dtype=np.complex64)


def build_model(*params):
    model = pb.Model(graphene.monolayer(), *params)
    model.report()


def test_decorator():
    pb.onsite_energy_modifier(lambda energy: energy)
    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda this_is_unexpected: None)
    assert "Unexpected argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda energy, x, y, z, w: None)
    assert "Unexpected argument" in str(excinfo.value)

    pb.onsite_energy_modifier(lambda energy: energy + 1)
    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda: 1)
    assert "Modifier must return numpy.ndarray" in str(excinfo.value)

    pb.site_position_modifier(lambda x, y, z: (x, y, z))
    with pytest.raises(RuntimeError) as excinfo:
        pb.site_position_modifier(lambda x, y, z: (x, y))
    assert "expected to return 3 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda: (one, one))
    assert "expected to return 1 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda x: np.zeros(x.size / 2))
    assert "must return the same shape" in str(excinfo.value)

    pb.hopping_energy_modifier(lambda energy: np.ones_like(energy, dtype=np.complex128))
    with pytest.raises(RuntimeError) as excinfo:
        pb.onsite_energy_modifier(lambda energy: np.ones_like(energy, dtype=np.complex128))
    assert "must not return complex" in str(excinfo.value)


@pb.site_state_modifier
def global_mod(state):
    return np.ones_like(state)


def test_callsig():
    assert "global_mod()" == str(global_mod)
    assert "global_mod()" == repr(global_mod)

    @pb.site_state_modifier
    def local_mod(state):
        return np.ones_like(state)
    assert "test_callsig()" == str(local_mod)
    assert "test_callsig()" == repr(local_mod)

    def wrapped_mod(a, b):
        @pb.site_state_modifier
        def actual_mod(state):
            return np.ones_like(state) * a * b
        return actual_mod
    assert "wrapped_mod(a=1, b=8)" == str(wrapped_mod(1, 8))
    assert "test_callsig.<locals>.wrapped_mod(a=1, b=8)" == repr(wrapped_mod(1, 8))


def test_cast():
    @pb.hopping_energy_modifier
    def complex_in_real_out(energy):
        return np.ones_like(energy, dtype=np.float64)

    assert np.isrealobj(complex_in_real_out(complex_one))
    assert np.iscomplexobj(complex_in_real_out.apply(complex_one, zero, zero, zero))
    assert not complex_in_real_out.is_complex()

    @pb.hopping_energy_modifier
    def real_in_complex_out(energy):
        return np.ones_like(energy, dtype=np.complex128)

    assert np.iscomplexobj(real_in_complex_out(complex_one))
    assert np.iscomplexobj(real_in_complex_out.apply(complex_one, zero, zero, zero))
    assert real_in_complex_out.is_complex()


def test_site_state():
    @pb.site_state_modifier
    def mod(state):
        return np.ones_like(state)
    assert np.all(mod(zero))
    assert np.all(mod.apply(zero, one, one, one, one))

    capture = []

    @pb.site_state_modifier
    def check_args(state, x, y, z, sub_id):
        capture[:] = (v.copy() for v in (state, x, y, z, sub_id))
        return state

    build_model(check_args)
    state, x, y, z, sub_id = capture
    assert np.all(state == [True, True])
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub_id, [0, 1])


def test_site_position():
    @pb.site_position_modifier
    def mod(x, y, z):
        return x + 1, y + 1, z + 1
    assert (one,) * 3 == mod(zero, zero, zero)
    assert (one,) * 3 == mod.apply(zero, zero, zero, one)

    capture = []

    @pb.site_position_modifier
    def check_args(x, y, z, sub_id):
        capture[:] = (v.copy() for v in (x, y, z, sub_id))
        return x, y, z

    build_model(check_args)
    x, y, z, sub_id = capture
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub_id, [0, 1])


def test_onsite():
    @pb.onsite_energy_modifier
    def mod(energy):
        return energy + 2
    assert np.all(2 == mod(zero))
    assert np.all(2 == mod.apply(zero, zero, zero, zero, one))

    capture = []

    @pb.onsite_energy_modifier
    def check_args(energy, x, y, z, sub_id):
        capture[:] = (v.copy() for v in (energy, x, y, z, sub_id))
        return energy

    build_model(check_args)
    energy, x, y, z, sub_id = capture
    assert np.allclose(energy, [0, 0])
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub_id, [0, 1])


def test_hopping_energy():
    @pb.hopping_energy_modifier
    def mod(energy):
        return energy * 2
    assert np.all(2 == mod(one))
    assert np.all(2 == mod.apply(one, zero, zero, zero, zero, zero, zero, zero))

    capture = []

    @pb.hopping_energy_modifier
    def check_args(energy, hop_id, x1, y1, z1, x2, y2, z2):
        capture[:] = (v.copy() for v in (energy, hop_id, x1, y1, z1, x2, y2, z2))
        return energy

    build_model(check_args)
    energy, hop_id, x1, y1, z1, x2, y2, z2 = capture
    assert np.allclose(energy, graphene.t)
    assert np.allclose(hop_id, 0)
    assert np.allclose(x1, 0)
    assert np.allclose(y1, -graphene.a_cc / 2)
    assert np.allclose(z1, 0)
    assert np.allclose(x2, 0)
    assert np.allclose(y2, graphene.a_cc / 2)
    assert np.allclose(z2, 0)


# Disabled for now. It doesn't work when the 'fast math' compiler flag is set.
def dont_test_invalid_return():
    @pb.onsite_energy_modifier
    def mod_inf(energy):
        return np.ones_like(energy) * np.inf

    with pytest.raises(RuntimeError) as excinfo:
        build_model(mod_inf)
    assert "NaN or INF" in str(excinfo.value)

    @pb.onsite_energy_modifier
    def mod_nan(energy):
        return np.ones_like(energy) * np.NaN

    with pytest.raises(RuntimeError) as excinfo:
        build_model(mod_nan)
    assert "NaN or INF" in str(excinfo.value)
