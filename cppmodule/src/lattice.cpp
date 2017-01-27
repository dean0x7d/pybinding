#include "Lattice.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_lattice(py::module& m) {
    py::class_<Hopping>(m, "Hopping")
        .def_readonly("relative_index", &Hopping::relative_index,
                      "Relative index between two unit cells - note that it may be [0, 0, 0]")
        .def_readonly("to_sublattice", &Hopping::to_sublattice,
                      "Sublattice ID of the hopping destination")
        .def_readonly("id", &Hopping::id,
                      "Points to the entry in :attr:`Lattice.hopping_energies`")
        .def_readonly("is_conjugate", &Hopping::is_conjugate,
                      "True if this is an automatically added complex conjugate")
        .def("__getstate__", [](Hopping const& h) {
            return py::make_tuple(h.relative_index, h.to_sublattice, h.id, h.is_conjugate);
        })
        .def("__setstate__", [](Hopping& h, py::tuple t) {
            new (&h) Hopping();
            h = {t[0].cast<decltype(h.relative_index)>(), t[1].cast<decltype(h.to_sublattice)>(),
                 t[2].cast<decltype(h.id)>(), t[3].cast<decltype(h.is_conjugate)>()};
        });

    py::class_<Sublattice>(m, "Sublattice")
        .def_readonly("position", &Sublattice::position, "Relative to global lattice offset")
        .def_readonly("alias", &Sublattice::alias,
                      "For supercells only: allows two sublattices to have the same ID")
        .def_readonly("hoppings", &Sublattice::hoppings, "List of :class:`~_pybinding.Hopping`")
        .def("__getstate__", [](Sublattice const& s) {
            return py::make_tuple(s.position, s.alias, s.hoppings);
        })
        .def("__setstate__", [](Sublattice& s, py::tuple t) {
            new (&s) Sublattice();
            s = {t[0].cast<decltype(s.position)>(), t[1].cast<decltype(s.alias)>(),
                 t[2].cast<decltype(s.hoppings)>()};
        });

    py::class_<Lattice::Sites>(m, "LatticeSites")
        .def_readonly("structure", &Lattice::Sites::structure)
        .def_readonly("energy", &Lattice::Sites::energy)
        .def_readonly("id", &Lattice::Sites::id)
        .def("__getstate__", [](Lattice::Sites const& s) {
            return py::make_tuple(s.structure, s.energy, s.id);
        })
        .def("__setstate__", [](Lattice::Sites& s, py::tuple t) {
            new (&s) Lattice::Sites();
            s = {t[0].cast<decltype(s.structure)>(), t[1].cast<decltype(s.energy)>(),
                 t[2].cast<decltype(s.id)>()};
        });

    py::class_<Lattice::Hoppings>(m, "LatticeHoppings")
        .def_readonly("structure", &Lattice::Hoppings::structure)
        .def_readonly("energy", &Lattice::Hoppings::energy)
        .def_readonly("id", &Lattice::Hoppings::id)
        .def("__getstate__", [](Lattice::Hoppings const& h) {
            return py::make_tuple(h.structure, h.energy, h.id);
        })
        .def("__setstate__", [](Lattice::Hoppings& h, py::tuple t) {
            new (&h) Lattice::Hoppings();
            h = {t[0].cast<decltype(h.structure)>(), t[1].cast<decltype(h.energy)>(),
                 t[2].cast<decltype(h.id)>()};
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
        .def_property_readonly("vectors", &Lattice::get_vectors)
        .def_property_readonly("sublattices",
                               [](Lattice const& l) { return l.get_sites().structure; })
        .def_property_readonly("sub_name_map",
                               [](Lattice const& l) { return l.get_sites().id; })
        .def_property_readonly("hopping_energies",
                               [](Lattice const& l) { return l.get_hoppings().energy; })
        .def_property_readonly("hop_name_map",
                               [](Lattice const& l) { return l.get_hoppings().id; })
        .def_property("offset", &Lattice::get_offset, &Lattice::set_offset)
        .def_property("min_neighbors", &Lattice::get_min_neighbors, &Lattice::set_min_neighbors)
        .def("__getstate__", [](Lattice const& l) {
            return py::make_tuple(l.get_vectors(), l.get_sites(), l.get_hoppings(),
                                  l.get_offset(), l.get_min_neighbors());
        })
        .def("__setstate__", [](Lattice& l, py::tuple t) {
            new (&l) Lattice(t[0].cast<Lattice::Vectors>(),
                             t[1].cast<Lattice::Sites>(),
                             t[2].cast<Lattice::Hoppings>());
            l.set_offset(t[3].cast<Cartesian>());
            l.set_min_neighbors(t[4].cast<int>());
        });
}
