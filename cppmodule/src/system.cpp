#include "system/System.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_system(py::module& m) {
    py::class_<CartesianArray>(m, "CartesianArray")
        .def_property_readonly("x", [](CartesianArray const& a) { return arrayref(a.x); })
        .def_property_readonly("y", [](CartesianArray const& a) { return arrayref(a.y); })
        .def_property_readonly("z", [](CartesianArray const& a) { return arrayref(a.z); })
        .def("__getstate__", [](CartesianArray const& a) {
            return py::make_tuple(arrayref(a.x), arrayref(a.y), arrayref(a.z));
        })
        .def("__setstate__", [](CartesianArray& a, py::tuple t) {
            using T = decltype(a.x);
            new (&a) CartesianArray(t[0].cast<T>(), t[1].cast<T>(), t[2].cast<T>());
        });

    py::class_<CompressedSublattices>(m, "CompressedSublattices")
        .def("__getstate__", [](CompressedSublattices const& c) {
            return py::dict("alias_ids"_a=c.get_alias_ids(), "site_counts"_a=c.get_site_counts());
        })
        .def("__setstate__", [](CompressedSublattices& c, py::dict d) {
            new (&c) CompressedSublattices(d["alias_ids"].cast<std::vector<storage_idx_t>>(),
                                           d["site_counts"].cast<std::vector<storage_idx_t>>());
        });

    using Boundary = System::Boundary;
    py::class_<Boundary>(m, "Boundary")
        .def_property_readonly("hoppings", [](Boundary const& b) { return csrref(b.hoppings); })
        .def_readonly("shift", &Boundary::shift)
        .def("__getstate__", [](Boundary const& b) { return py::make_tuple(b.hoppings, b.shift); })
        .def("__setstate__", [](Boundary& b, py::tuple t) {
            new (&b) Boundary{t[0].cast<decltype(b.hoppings)>(), t[1].cast<decltype(b.shift)>()};
        });

    py::class_<System, std::shared_ptr<System>>(m, "System")
        .def("find_nearest", &System::find_nearest, "position"_a, "sublattice"_a="")
        .def_readonly("lattice", &System::lattice)
        .def_readonly("positions", &System::positions)
        .def_property_readonly("sublattices", [](System const& s) {
            return s.compressed_sublattices.decompress();
        })
        .def_property_readonly("hoppings", [](System const& s) { return csrref(s.hoppings); })
        .def_readonly("boundaries", &System::boundaries)
        .def_readonly("has_unbalanced_hoppings", &System::has_unbalanced_hoppings)
        .def("__getstate__", [](System const& s) {
            return py::dict("lattice"_a=s.lattice, "positions"_a=s.positions,
                            "compressed_sublattices"_a=s.compressed_sublattices,
                            "hoppings"_a=s.hoppings, "boundaries"_a=s.boundaries,
                            "has_unbalanced_hoppings"_a=s.has_unbalanced_hoppings);
        })
        .def("__setstate__", [](System& s, py::dict d) {
            new (&s) System(d["lattice"].cast<decltype(s.lattice)>());
            s.positions = d["positions"].cast<decltype(s.positions)>();
            s.compressed_sublattices =
                d["compressed_sublattices"].cast<decltype(s.compressed_sublattices)>();
            s.hoppings = d["hoppings"].cast<decltype(s.hoppings)>();
            s.boundaries = d["boundaries"].cast<decltype(s.boundaries)>();
            s.has_unbalanced_hoppings =
                d["has_unbalanced_hoppings"].cast<decltype(s.has_unbalanced_hoppings)>();
        });
}
