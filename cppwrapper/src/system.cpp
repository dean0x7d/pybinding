#include "system/System.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/pure_virtual.hpp>

using namespace boost::python;
using namespace tbm;

class PySiteStateModifier : public SiteStateModifier,
                            public wrapper<SiteStateModifier> {
public:
    PySiteStateModifier(object py_apply, int min_neighbors)
        : SiteStateModifier({}, min_neighbors) {
        apply = [py_apply](ArrayX<bool>& state, CartesianArray const& p, SubIdRef sub) {
            object result = py_apply(arrayref(state), arrayref(p.x), arrayref(p.y),
                                     arrayref(p.z), sub);
            extract_array(state, result);
        };
    }
};

class PyPositionModifier : public PositionModifier,
                           public wrapper<PositionModifier> {
public:
    PyPositionModifier(object py_apply) : PositionModifier({}) {
        apply = [py_apply](CartesianArray& p, SubIdRef sub) {
            object result = py_apply(arrayref(p.x), arrayref(p.y), arrayref(p.z), sub);
            extract_array(p.x, result[0]);
            extract_array(p.y, result[1]);
            extract_array(p.z, result[2]);
        };
    }
};

class PyShape : public tbm::Shape, public wrapper<tbm::Shape> {
public:
    PyShape(Vertices const& vertices, object py_contains, Cartesian offset)
        : Shape(vertices, {}, offset) {
        contains = [py_contains](CartesianArray const& p) {
            object result = py_contains(arrayref(p.x), arrayref(p.y), arrayref(p.z));
            return extract<ArrayX<bool>>(result)();
        };
    }
};

class PyHoppingGenerator : public HoppingGenerator,
                           public wrapper<HoppingGenerator> {
public:
    PyHoppingGenerator(std::string const& name, std::complex<double> energy, object py_make)
        : HoppingGenerator(name, energy, {}) {
        make = [py_make](CartesianArray const& p, SubIdRef sublattice) {
            object py_result = py_make(arrayref(p.x), arrayref(p.y), arrayref(p.z), sublattice);
            auto result = PyHoppingGenerator::Result{ArrayXi(), ArrayXi()};
            extract_array(result.from, py_result[0]);
            extract_array(result.to, py_result[1]);
            return result;
        };
    }
};

void export_system() {
    using Boundary = System::Boundary;
    class_<Boundary>{"Boundary"}
    .add_property("hoppings", internal_ref(&Boundary::hoppings))
    .add_property("shift", copy_value(&Boundary::shift))
    .enable_pickling()
    .def("__getstate__", [](object o) { return make_tuple(o.attr("hoppings"), o.attr("shift")); })
    .def("__setstate__", [](Boundary& b, tuple t) {
        b = {extract<decltype(b.hoppings)>(t[0]), extract<decltype(b.shift)>(t[1])};
    })
    ;

    using Port = System::Port;
    class_<Port>{"Port"}
    .add_property("shift", copy_value(&Port::shift))
    .add_property("indices", copy_value(&Port::indices))
    .add_property("outer_hoppings", internal_ref(&Port::outer_hoppings))
    .add_property("inner_hoppings", internal_ref(&Port::inner_hoppings))
    .enable_pickling()
    .def("__getstate__", [](object o) {
        return make_tuple(o.attr("shift"), o.attr("indices"),
                          o.attr("outer_hoppings"), o.attr("inner_hoppings"));
    })
    .def("__setstate__", [](Port& x, tuple t) {
        x.shift = extract<decltype(x.shift)>(t[0]);
        x.indices = extract<decltype(x.indices)>(t[1]);
        x.outer_hoppings = extract<decltype(x.outer_hoppings)>(t[2]);
        x.inner_hoppings = extract<decltype(x.inner_hoppings)>(t[3]);
    })
    ;

    class_<System, std::shared_ptr<System>, noncopyable>{"System", init<Lattice const&>()}
    .def("find_nearest", &System::find_nearest, args("self", "position", "sublattice"_kw=-1))
    .add_property("lattice", extended(&System::lattice, "Lattice"))
    .add_property("positions", internal_ref(&System::positions))
    .add_property("sublattices", dense_uref(&System::sublattices))
    .add_property("hoppings", internal_ref(&System::hoppings))
    .add_property("boundaries", &System::boundaries)
    .add_property("ports", &System::ports)
    .def_readonly("has_unbalanced_hoppings", &System::has_unbalanced_hoppings)
    .enable_pickling()
    .def("__getinitargs__", [](System const& s) { return make_tuple(s.lattice); })
    .def("__getstate__", [](object s) {
        return make_tuple(s.attr("positions"), s.attr("sublattices"), s.attr("hoppings"),
                          s.attr("boundaries"), s.attr("ports"), s.attr("has_unbalanced_hoppings"));
    })
    .def("__setstate__", [](System& s, tuple t) {
        s.positions = extract<decltype(s.positions)>(t[0]);
        extract_array(s.sublattices, t[1]);
        s.hoppings = extract<decltype(s.hoppings)>(t[2]);
        s.boundaries = extract<decltype(s.boundaries)>(t[3]);
        s.ports = extract<decltype(s.ports)>(t[4]);
        s.has_unbalanced_hoppings = extract<decltype(s.has_unbalanced_hoppings)>(t[5]);
    })
    ;

    using tbm::Hopping;
    class_<Hopping>{"Hopping"}
    .add_property("relative_index", copy_value(&Hopping::relative_index))
    .def_readonly("to_sublattice", &Hopping::to_sublattice)
    .def_readonly("id", &Hopping::id)
    .def_readonly("is_conjugate", &Hopping::is_conjugate)
    .enable_pickling()
    .def("__getstate__", [](Hopping const& h) {
        return make_tuple(h.relative_index, h.to_sublattice, h.id, h.is_conjugate);
    })
    .def("__setstate__", [](Hopping& h, tuple t) {
        h = {extract<decltype(h.relative_index)>(t[0]), extract<decltype(h.to_sublattice)>(t[1]),
             extract<decltype(h.id)>(t[2]), extract<decltype(h.is_conjugate)>(t[3])};
    })
    ;

    using tbm::Sublattice;
    class_<Sublattice>{"Sublattice"}
    .add_property("offset", copy_value(&Sublattice::offset))
    .def_readonly("onsite", &Sublattice::onsite)
    .def_readonly("alias", &Sublattice::alias)
    .add_property("hoppings", &Sublattice::hoppings)
    .enable_pickling()
    .def("__getstate__", [](Sublattice const& s) {
        return make_tuple(s.offset, s.onsite, s.alias, s.hoppings);
    })
    .def("__setstate__", [](Sublattice& s, tuple t) {
        s = {extract<decltype(s.offset)>(t[0]), extract<decltype(s.onsite)>(t[1]),
             extract<decltype(s.alias)>(t[2]), extract<decltype(s.hoppings)>(t[3])};
    })
    ;

    class_<SubIdRef>{"SubIdRef", no_init}
    .add_property("ids", internal_ref([](SubIdRef const& s) { return arrayref(s.ids); }))
    .add_property("name_map", copy_value([](SubIdRef const& s) { return s.name_map; }))
    ;

    class_<Lattice>{"Lattice",
                    init<Cartesian, optional<Cartesian, Cartesian>>(args("a1", "a2", "a3"))}
    .def("_add_sublattice", &Lattice::add_sublattice,
         args("self", "name", "offset", "onsite_potential"_kw=.0f, "alias"_kw=-1))
    .def("_add_hopping", &Lattice::add_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "energy"))
    .def("_register_hopping_energy", &Lattice::register_hopping_energy, args("self", "energy"))
    .def("_add_registered_hopping", &Lattice::add_registered_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "id"))
    .add_property("vectors", &Lattice::vectors)
    .add_property("sublattices", &Lattice::sublattices)
    .add_property("hopping_energies", &Lattice::hopping_energies)
    .add_property("sub_name_map", &Lattice::sub_name_map)
    .add_property("hop_name_map", &Lattice::hop_name_map)
    .def_readwrite("min_neighbors", &Lattice::min_neighbours)
    .enable_pickling()
    .def("__getinitargs__", [](Lattice const& l) { return l.vectors; })
    .def("__getstate__", [](Lattice const& l) {
        return make_tuple(l.sublattices, l.hopping_energies, l.sub_name_map, l.hop_name_map,
                          l.min_neighbours);
    })
    .def("__setstate__", [](Lattice& l, tuple t) {
        l.sublattices = extract<decltype(l.sublattices)>(t[0]);
        l.has_onsite_energy = std::any_of(l.sublattices.begin(), l.sublattices.end(),
                                          [](Sublattice const& sub) { return sub.onsite != 0; });
        l.hopping_energies = extract<decltype(l.hopping_energies)>(t[1]);
        l.has_complex_hopping = std::any_of(l.hopping_energies.begin(), l.hopping_energies.end(),
                                            [](std::complex<double> e) { return e.imag() != .0; });
        l.sub_name_map = extract<decltype(l.sub_name_map)>(t[2]);
        l.hop_name_map = extract<decltype(l.hop_name_map)>(t[3]);
        l.min_neighbours = extract<decltype(l.min_neighbours)>(t[4]);
    })
    ;

    class_<tbm::Primitive> {
        "Primitive", "Shape of the primitive unit cell",
        init<int, int, int> {args("self", "a1"_kw=1, "a2"_kw=1, "a3"_kw=1)}
    };

    class_<PyShape, noncopyable>{"Shape",
        init<PyShape::Vertices const&, object, Cartesian>{
            args("self", "vertices", "contains", "offset")
        }
    }
    .add_property("vertices", copy_value(&PyShape::vertices))
    .add_property("offset", copy_value(&PyShape::offset))
    ;

    class_<tbm::Line, bases<tbm::Shape>, noncopyable>{"Line",
        init<Cartesian, Cartesian, optional<Cartesian>>{args("self", "a", "b", "offset")}
    };

    using tbm::Polygon;
    class_<Polygon, bases<tbm::Shape>, noncopyable> {"Polygon",
        init<Polygon::Vertices const&, Cartesian> {args("self", "vertices", "offset")}
    };

    class_<tbm::Symmetry>{"TranslationalSymmetry", init<Cartesian>{args("self", "length")}};

    class_<PySiteStateModifier, noncopyable>{
        "SiteStateModifier",
        init<object, int>(args("self", "apply", "min_neighbors"_kw=0))
    };
    class_<PyPositionModifier, noncopyable>{
        "PositionModifier",
        init<object>(args("self", "apply"))
    };

    class_<PyHoppingGenerator, noncopyable>{
        "HoppingGenerator",
        init<std::string const&, std::complex<double>, object>{
            args("self", "name", "energy", "make")
        }
    };
}
