import numpy as np

from _pybinding import wrapper_tests


def test_variant_caster():
    assert wrapper_tests.variant_cast() == (5, "Hello")
    assert wrapper_tests.variant_load(1) == "int"
    assert wrapper_tests.variant_load("1") == "std::string"


def test_array_ref():
    r = wrapper_tests.TestArrayRef()

    assert r.a.shape == (2, 2, 3)
    assert r.a[0, :].shape == (2, 3)
    assert r.a[0].shape == (2, 3)
    assert np.all(r.a[0] == [[0, 1, 2], [3, 4, 5]])
    assert np.all(r.a[1] == [[6, 7, 8], [9, 10, 11]])
