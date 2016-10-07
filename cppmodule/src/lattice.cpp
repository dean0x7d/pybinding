#include "system/Lattice.hpp"
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
        .def_readonly("offset", &Sublattice::offset, "Relative to global lattice offset")
        .def_readonly("onsite", &Sublattice::onsite, "Onsite energy")
        .def_readonly("alias", &Sublattice::alias,
                      "For supercells only: allows two sublattices to have the same ID")
        .def_readonly("hoppings", &Sublattice::hoppings, "List of :class:`~_pybinding.Hopping`")
        .def("__getstate__", [](Sublattice const& s) {
            return py::make_tuple(s.offset, s.onsite, s.alias, s.hoppings);
        })
        .def("__setstate__", [](Sublattice& s, py::tuple t) {
            new (&s) Sublattice();
            s = {t[0].cast<decltype(s.offset)>(), t[1].cast<decltype(s.onsite)>(),
                 t[2].cast<decltype(s.alias)>(), t[3].cast<decltype(s.hoppings)>()};
        });

    py::class_<Lattice>(m, "Lattice")
        .def(py::init<Cartesian, Cartesian, Cartesian>(),
             "a1"_a, "a2"_a=Cartesian{0, 0, 0}, "a3"_a=Cartesian{0, 0, 0})
        .def("add_sublattice", &Lattice::add_sublattice,
             "name"_a, "offset"_a, "onsite_potential"_a, "alias"_a)
        .def("add_hopping", &Lattice::add_hopping,
             "relative_index"_a, "from_sublattice"_a, "to_sublattice"_a, "energy"_a)
        .def("register_hopping_energy", &Lattice::register_hopping_energy, "name"_a, "energy"_a)
        .def("add_registered_hopping", &Lattice::add_registered_hopping,
             "relative_index"_a, "from_sublattice"_a, "to_sublattice"_a, "id"_a)
        .def_readonly("vectors", &Lattice::vectors)
        .def_readonly("sublattices", &Lattice::sublattices)
        .def_readonly("hopping_energies", &Lattice::hopping_energies)
        .def_readonly("sub_name_map", &Lattice::sub_name_map)
        .def_readonly("hop_name_map", &Lattice::hop_name_map)
        .def_property("offset", [](Lattice const& l) { return l.offset; }, &Lattice::set_offset)
        .def_readwrite("min_neighbors", &Lattice::min_neighbors)
        .def("__getstate__", [](Lattice const& l) {
            return py::make_tuple(l.vectors, l.sublattices, l.hopping_energies, l.sub_name_map,
                                  l.hop_name_map, l.offset, l.min_neighbors);
        })
        .def("__setstate__", [](Lattice& l, py::tuple t) {
            new (&l) Lattice(t[0].cast<decltype(l.vectors)>());
            l.sublattices = t[1].cast<decltype(l.sublattices)>();
            l.has_onsite_energy = std::any_of(l.sublattices.begin(), l.sublattices.end(),
                                              [](Sublattice const& sub) { return sub.onsite != 0; });
            l.hopping_energies = t[2].cast<decltype(l.hopping_energies)>();
            l.has_complex_hopping = std::any_of(l.hopping_energies.begin(), l.hopping_energies.end(),
                                                [](std::complex<double> e) { return e.imag() != .0; });
            l.sub_name_map = t[3].cast<decltype(l.sub_name_map)>();
            l.hop_name_map = t[4].cast<decltype(l.hop_name_map)>();
            l.offset = t[5].cast<decltype(l.offset)>();
            l.min_neighbors = t[6].cast<decltype(l.min_neighbors)>();
        });
}
