import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


one, zero = np.ones(1), np.zeros(1)
complex_one = np.ones(1, dtype=np.complex64)


def build_model(*params):
    model = pb.Model(graphene.lattice.monolayer(), *params)
    model.report()


def test_decorator():
    pb.modifier.onsite_energy(lambda potential: potential)
    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.onsite_energy(lambda this_is_unexpected: None)
    assert "Unexpected argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.onsite_energy(lambda potential, x, y, z, w: None)
    assert "Unexpected argument" in str(excinfo.value)

    pb.modifier.onsite_energy(lambda potential: potential + 1)
    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.onsite_energy(lambda: 1)
    assert "Modifier must return numpy.ndarray" in str(excinfo.value)

    pb.modifier.site_position(lambda x, y, z: (x, y, z))
    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.site_position(lambda x, y, z: (x, y))
    assert "expected to return 3 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.onsite_energy(lambda: (one, one))
    assert "expected to return 1 ndarray(s), but got 2" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.onsite_energy(lambda x: np.zeros(x.size / 2))
    assert "must return the same shape" in str(excinfo.value)

    pb.modifier.onsite_energy(lambda potential: np.ones_like(potential, dtype=np.complex128))
    with pytest.raises(RuntimeError) as excinfo:
        pb.modifier.site_state(lambda state: np.ones_like(state, dtype=np.complex128))
    assert "must not return complex" in str(excinfo.value)


@pb.modifier.site_state
def global_mod(state):
    return np.ones_like(state)


def test_callsig():
    assert "global_mod()" == str(global_mod)
    assert "global_mod()" == repr(global_mod)

    @pb.modifier.site_state
    def local_mod(state):
        return np.ones_like(state)
    assert "test_callsig()" == str(local_mod)
    assert "test_callsig()" == repr(local_mod)

    def wrapped_mod(a, b):
        @pb.modifier.site_state
        def actual_mod(state):
            return np.ones_like(state) * a * b
        return actual_mod
    assert "wrapped_mod(a=1, b=8)" == str(wrapped_mod(1, 8))
    assert "test_callsig.<locals>.wrapped_mod(a=1, b=8)" == repr(wrapped_mod(1, 8))


def test_cast():
    @pb.modifier.onsite_energy
    def complex_in_real_out(potential):
        return np.ones_like(potential, dtype=np.float64)

    assert np.isrealobj(complex_in_real_out(complex_one))
    assert np.iscomplexobj(complex_in_real_out.apply(complex_one, zero, zero, zero))
    assert not complex_in_real_out.is_complex()

    @pb.modifier.onsite_energy
    def real_in_complex_out(potential):
        return np.ones_like(potential, dtype=np.complex128)

    assert np.iscomplexobj(real_in_complex_out(complex_one))
    assert np.iscomplexobj(real_in_complex_out.apply(complex_one, zero, zero, zero))
    assert real_in_complex_out.is_complex()


def test_site_state():
    @pb.modifier.site_state
    def mod(state):
        return np.ones_like(state)
    assert np.all(mod(zero))
    assert np.all(mod.apply(zero, one, one, one, one))

    capture = []

    @pb.modifier.site_state
    def check_args(state, x, y, z, sub):
        capture[:] = (v.copy() for v in (state, x, y, z, sub))
        return state

    build_model(check_args)
    state, x, y, z, sub = capture
    assert np.all(state == [True, True])
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub, [0, 1])


def test_site_position():
    @pb.modifier.site_position
    def mod(x, y, z):
        return x + 1, y + 1, z + 1
    assert (one,) * 3 == mod(zero, zero, zero)
    assert (one,) * 3 == mod.apply(zero, zero, zero, one)

    capture = []

    @pb.modifier.site_position
    def check_args(x, y, z, sub):
        capture[:] = (v.copy() for v in (x, y, z, sub))
        return x, y, z

    build_model(check_args)
    x, y, z, sub = capture
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub, [0, 1])


def test_onsite():
    @pb.modifier.onsite_energy
    def mod(potential):
        return potential + 2
    assert np.all(2 == mod(zero))
    assert np.all(2 == mod.apply(zero, zero, zero, zero, one))

    capture = []

    @pb.modifier.onsite_energy
    def check_args(potential, x, y, z, sub):
        capture[:] = (v.copy() for v in (potential, x, y, z, sub))
        return potential

    build_model(check_args)
    potential, x, y, z, sub = capture
    assert np.allclose(potential, [0, 0])
    assert np.allclose(x, [0, 0])
    assert np.allclose(y, [-graphene.a_cc / 2, graphene.a_cc / 2])
    assert np.allclose(z, [0, 0])
    assert np.allclose(sub, [0, 1])


def test_hopping_energy():
    @pb.modifier.hopping_energy
    def mod(hopping):
        return hopping * 2
    assert np.all(2 == mod(one))
    assert np.all(2 == mod.apply(one, zero, zero, zero, zero, zero, zero, zero))

    capture = []

    @pb.modifier.hopping_energy
    def check_args(hopping, hop_id, x1, y1, z1, x2, y2, z2):
        capture[:] = (v.copy() for v in (hopping, hop_id, x1, y1, z1, x2, y2, z2))
        return hopping

    build_model(check_args)
    hopping, hop_id, x1, y1, z1, x2, y2, z2 = capture
    assert np.allclose(hopping, graphene.t)
    assert np.allclose(hop_id, 0)
    assert np.allclose(x1, 0)
    assert np.allclose(y1, -graphene.a_cc / 2)
    assert np.allclose(z1, 0)
    assert np.allclose(x2, 0)
    assert np.allclose(y2, graphene.a_cc / 2)
    assert np.allclose(z2, 0)
