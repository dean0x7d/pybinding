#include "Lattice.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_lattice(py::module& m) {
    py::class_<Lattice::Sublattice>(m, "Sublattice")
        .def_readonly("position", &Lattice::Sublattice::position,
                      "Relative to global lattice offset")
        .def_readonly("energy", &Lattice::Sublattice::energy, "Onsite energy matrix")
        .def_property_readonly("unique_id",
                               [](Lattice::Sublattice const& s) { return s.unique_id.value(); },
                               "Different for each sublattice")
        .def_property_readonly("alias_id",
                               [](Lattice::Sublattice const& s) { return s.alias_id.value(); },
                               "For supercells only: indicates sublattices which may be aliased")
        .def("__getstate__", [](Lattice::Sublattice const& s) {
            return py::dict("position"_a=s.position, "energy"_a=s.energy,
                            "unique_id"_a= s.unique_id.value(), "alias_id"_a= s.alias_id.value());
        })
        .def("__setstate__", [](Lattice::Sublattice& s, py::dict d) {
            new (&s) Lattice::Sublattice();
            s = {d["position"].cast<decltype(s.position)>(),
                 d["energy"].cast<decltype(s.energy)>(),
                 SubID(d["unique_id"].cast<storage_idx_t>()),
                 SubAliasID(d["alias_id"].cast<storage_idx_t>())};
        });

    py::class_<Lattice::HoppingTerm>(m, "HoppingTerm")
        .def_readonly("relative_index", &Lattice::HoppingTerm::relative_index,
                      "Relative index between two unit cells - note that it may be [0, 0, 0]")
        .def_property_readonly("from", [](Lattice::HoppingTerm const& h) { return h.from.value();},
                               "Sublattice ID of the source")
        .def_property_readonly("to", [](Lattice::HoppingTerm const& h) { return h.to.value();},
                               "Sublattice ID of the destination")
        .def("__getstate__", [](Lattice::HoppingTerm const& h) {
            return py::dict("relative_index"_a=h.relative_index, "from"_a= h.from.value(),
                            "to"_a= h.to.value());
        })
        .def("__setstate__", [](Lattice::HoppingTerm& s, py::dict d) {
            new (&s) Lattice::HoppingTerm();
            s = {d["relative_index"].cast<decltype(s.relative_index)>(),
                 SubID(d["from"].cast<storage_idx_t>()),
                 SubID(d["to"].cast<storage_idx_t>())};
        });

    py::class_<Lattice::HoppingFamily>(m, "HoppingFamily")
        .def_readonly("energy", &Lattice::HoppingFamily::energy, "Hopping energy matrix")
        .def_property_readonly("family_id", [](Lattice::HoppingFamily const& f) {
            return f.family_id.value();
        })
        .def_readonly("terms", &Lattice::HoppingFamily::terms,
                      "List of :class:`~_pybinding.HoppingTerm`")
        .def("__getstate__", [](Lattice::HoppingFamily const& f) {
            return py::dict("energy"_a=f.energy, "family_id"_a= f.family_id.value(),
                            "terms"_a=f.terms);
        })
        .def("__setstate__", [](Lattice::HoppingFamily& s, py::dict d) {
            new (&s) Lattice::HoppingFamily();
            s = {d["energy"].cast<decltype(s.energy)>(),
                 HopID(d["family_id"].cast<storage_idx_t>()),
                 d["terms"].cast<decltype(s.terms)>()};
        });

    py::class_<Lattice>(m, "Lattice")
        .def(py::init<Cartesian, Cartesian, Cartesian>(),
             "a1"_a, "a2"_a=Cartesian{0, 0, 0}, "a3"_a=Cartesian{0, 0, 0})
        .def("add_sublattice",
             &Lattice::add_sublattice | resolve<string_view, Cartesian, double>())
        .def("add_sublattice",
             &Lattice::add_sublattice | resolve<string_view, Cartesian, VectorXd const&>())
        .def("add_sublattice",
             &Lattice::add_sublattice | resolve<string_view, Cartesian, MatrixXcd const&>())
        .def("add_alias", &Lattice::add_alias)
        .def("register_hopping_energy",
             &Lattice::register_hopping_energy | resolve<string_view, std::complex<double>>())
        .def("register_hopping_energy",
             &Lattice::register_hopping_energy | resolve<string_view, MatrixXcd const&>())
        .def("add_hopping",
             &Lattice::add_hopping | resolve<Index3D, string_view, string_view, string_view>())
        .def("add_hopping",
             &Lattice::add_hopping | resolve<Index3D, string_view, string_view, std::complex<double>>())
        .def("add_hopping",
             &Lattice::add_hopping | resolve<Index3D, string_view, string_view, MatrixXcd const&>())
        .def_property_readonly("ndim", &Lattice::ndim)
        .def_property_readonly("nsub", &Lattice::nsub)
        .def_property_readonly("vectors", &Lattice::get_vectors)
        .def_property_readonly("sublattices", &Lattice::get_sublattices)
        .def_property_readonly("hoppings", &Lattice::get_hoppings)
        .def_property_readonly("sub_name_map", &Lattice::sub_name_map)
        .def_property_readonly("hop_name_map", &Lattice::hop_name_map)
        .def_property("offset", &Lattice::get_offset, &Lattice::set_offset)
        .def_property("min_neighbors", &Lattice::get_min_neighbors, &Lattice::set_min_neighbors)
        .def("__getstate__", [](Lattice const& l) {
            return py::dict("vectors"_a=l.get_vectors(),
                            "sublattices"_a=l.get_sublattices(),
                            "hoppings"_a=l.get_hoppings(),
                            "offset"_a=l.get_offset(),
                            "min_neighbors"_a=l.get_min_neighbors());
        })
        .def("__setstate__", [](Lattice& l, py::dict d) {
            new (&l) Lattice(d["vectors"].cast<Lattice::Vectors>(),
                             d["sublattices"].cast<Lattice::Sublattices>(),
                             d["hoppings"].cast<Lattice::Hoppings>());
            l.set_offset(d["offset"].cast<Cartesian>());
            l.set_min_neighbors(d["min_neighbors"].cast<int>());
        });
}
