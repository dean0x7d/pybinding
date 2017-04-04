#include "Lattice.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_lattice(py::module& m) {
    using Sub = Lattice::Sublattice;
    py::class_<Sub>(m, "Sublattice")
        .def_readonly("position", &Sub::position, "Relative to global lattice offset")
        .def_readonly("energy", &Sub::energy, "Onsite energy matrix")
        .def_readonly("unique_id", &Sub::unique_id, "Different for each sublattice")
        .def_readonly("alias_id", &Sub::alias_id, "May be shared by multiple (e.g. supercells)")
        .def("__getstate__", [](Sub const& s) {
            return py::dict("position"_a=s.position, "energy"_a=s.energy,
                            "unique_id"_a=s.unique_id, "alias_id"_a=s.alias_id);
        })
        .def("__setstate__", [](Sub& s, py::dict d) {
            new (&s) Sub{d["position"].cast<decltype(s.position)>(),
                         d["energy"].cast<decltype(s.energy)>(),
                         d["unique_id"].cast<decltype(s.unique_id)>(),
                         d["alias_id"].cast<decltype(s.alias_id)>()};
        });

    using HT = Lattice::HoppingTerm;
    py::class_<HT>(m, "HoppingTerm")
        .def_readonly("relative_index", &HT::relative_index,
                      "Relative index between two unit cells - note that it may be [0, 0, 0]")
        .def_readonly("from_id", &HT::from, "Sublattice ID of the source")
        .def_readonly("to_id", &HT::to, "Sublattice ID of the destination")
        .def("__getstate__", [](HT const& h) {
            return py::dict("relative_index"_a=h.relative_index, "from"_a=h.from, "to"_a=h.to);
        })
        .def("__setstate__", [](HT& h, py::dict d) {
            new (&h) HT{d["relative_index"].cast<decltype(h.relative_index)>(),
                        d["from"].cast<decltype(h.from)>(),
                        d["to"].cast<decltype(h.to)>()};
        });

    using HF = Lattice::HoppingFamily;
    py::class_<HF>(m, "HoppingFamily")
        .def_readonly("energy", &HF::energy, "Hopping matrix shared by all terms in this family")
        .def_readonly("family_id", &HF::family_id, "Different for each family")
        .def_readonly("terms", &HF::terms, "List of :class:`~_pybinding.HoppingTerm`")
        .def("__getstate__", [](HF const& f) {
            return py::dict("energy"_a=f.energy, "family_id"_a=f.family_id, "terms"_a=f.terms);
        })
        .def("__setstate__", [](HF& f, py::dict d) {
            new (&f) HF{d["energy"].cast<decltype(f.energy)>(),
                        d["family_id"].cast<decltype(f.family_id)>(),
                        d["terms"].cast<decltype(f.terms)>()};
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
        .def_property_readonly("nhop", &Lattice::nhop)
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
