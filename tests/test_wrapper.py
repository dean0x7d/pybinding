from _pybinding import wrapper_tests


def test_variant_caster():
    assert wrapper_tests.variant_cast() == (5, "Hello")
    assert wrapper_tests.variant_load(1) == "int"
    assert wrapper_tests.variant_load("1") == "std::string"
