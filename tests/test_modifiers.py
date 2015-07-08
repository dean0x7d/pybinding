import pytest

import numpy as np
import pybinding as pb


one, zero = np.ones(1), np.zeros(1)
complex_one = np.ones(1, dtype=np.complex64)


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
    assert np.all(mod.apply(zero, one, one, one))


def test_site_position():
    @pb.modifier.site_position
    def mod(x, y, z):
        return x + 1, y + 1, z + 1
    assert (one,) * 3 == mod(zero, zero, zero)
    assert (one,) * 3 == mod.apply(zero, zero, zero)


def test_onsite():
    @pb.modifier.onsite_energy
    def mod(potential):
        return potential + 2
    assert np.all(2 == mod(zero))
    assert np.all(2 == mod.apply(zero, zero, zero, zero))


def test_hopping_energy():
    @pb.modifier.hopping_energy
    def mod(hopping):
        return hopping * 2
    assert np.all(2 == mod(one))
    assert np.all(2 == mod.apply(one, zero, zero, zero, zero, zero, zero))
