#include "wrappers.hpp"

namespace {
    struct TestArrayRef {
        Eigen::ArrayXi a = Eigen::ArrayXi::LinSpaced(12, 0, 12);
    };
}

void wrapper_tests(py::module& pm) {
    auto m = pm.def_submodule("wrapper_tests");

    m.def("variant_load", [](cpb::var::variant<int, std::string> v) {
        return v.is<int>() ? "int" : "std::string";
    });

    m.def("variant_cast", []() {
        using V = cpb::var::variant<int, std::string>;
        return py::make_tuple(V(5), V("Hello"));
    });

    py::class_<TestArrayRef>(m, "TestArrayRef")
        .def(py::init<>())
        .def_property_readonly("a", [](TestArrayRef const& r) {
            return cpb::num::arrayref(r.a.data(), 2, 2, 3);
        });
}
