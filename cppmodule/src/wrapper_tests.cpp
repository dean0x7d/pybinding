#include "wrappers.hpp"

void wrapper_tests(py::module& pm) {
    auto m = pm.def_submodule("wrapper_tests");

    m.def("variant_load", [](cpb::var::variant<int, std::string> v) {
        return v.is<int>() ? "int" : "std::string";
    });

    m.def("variant_cast", []() {
        using V = cpb::var::variant<int, std::string>;
        return py::make_tuple(V(5), V("Hello"));
    });
}
