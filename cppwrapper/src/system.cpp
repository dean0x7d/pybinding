#include "system/System.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/pure_virtual.hpp>

using namespace boost::python;
using namespace cpb;

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

struct PyContains {
    object contains;

    ArrayX<bool> operator()(CartesianArray const& p) const {
        object result = contains(arrayref(p.x), arrayref(p.y), arrayref(p.z));
        return extract<ArrayX<bool>>(result)();
    }
};

class PyShape : public Shape, public wrapper<Shape> {
public:
    PyShape(Vertices const& vertices, object contains) : Shape(vertices, PyContains{contains}) {}
};

class PyFreeformShape : public FreeformShape, public wrapper<FreeformShape> {
public:
    PyFreeformShape(object contains, Cartesian width, Cartesian center)
        : FreeformShape(PyContains{contains}, width, center) {}
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
    .add_property("hoppings", return_internal_copy(&Boundary::hoppings))
    .add_property("shift", return_copy(&Boundary::shift))
    .enable_pickling()
    .def("__getstate__", [](object o) { return make_tuple(o.attr("hoppings"), o.attr("shift")); })
    .def("__setstate__", [](Boundary& b, tuple t) {
        b = {extract<decltype(b.hoppings)>(t[0]), extract<decltype(b.shift)>(t[1])};
    })
    ;

    class_<System, std::shared_ptr<System>, noncopyable>{"System", init<Lattice const&>()}
    .def("find_nearest", &System::find_nearest, args("self", "position", "sublattice"_kw=-1))
    .add_property("lattice", extended(&System::lattice, "Lattice"))
    .add_property("positions", return_reference(&System::positions))
    .add_property("sublattices", return_arrayref(&System::sublattices))
    .add_property("hoppings", return_internal_copy(&System::hoppings))
    .add_property("boundaries", &System::boundaries)
    .def_readonly("has_unbalanced_hoppings", &System::has_unbalanced_hoppings)
    .enable_pickling()
    .def("__getinitargs__", [](System const& s) { return make_tuple(s.lattice); })
    .def("__getstate__", [](object s) {
        return make_tuple(s.attr("positions"), s.attr("sublattices"), s.attr("hoppings"),
                          s.attr("boundaries"), 0, s.attr("has_unbalanced_hoppings"));
    })
    .def("__setstate__", [](System& s, tuple t) {
        s.positions = extract<decltype(s.positions)>(t[0]);
        extract_array(s.sublattices, t[1]);
        s.hoppings = extract<decltype(s.hoppings)>(t[2]);
        s.boundaries = extract<decltype(s.boundaries)>(t[3]);
        s.has_unbalanced_hoppings = extract<decltype(s.has_unbalanced_hoppings)>(t[5]);
    })
    ;

    class_<Hopping>{"Hopping"}
    .add_property("relative_index", return_copy(&Hopping::relative_index),
                  "Relative index between two unit cells - note that it may be (0, 0, 0)")
    .def_readonly("to_sublattice", &Hopping::to_sublattice,
                  "Sublattice ID of the hopping destination")
    .def_readonly("id", &Hopping::id, "Points to the entry in :attr:`Lattice.hopping_energies`")
    .def_readonly("is_conjugate", &Hopping::is_conjugate,
                  "True if this is an automatically added complex conjugate")
    .enable_pickling()
    .def("__getstate__", [](Hopping const& h) {
        return make_tuple(h.relative_index, h.to_sublattice, h.id, h.is_conjugate);
    })
    .def("__setstate__", [](Hopping& h, tuple t) {
        h = {extract<decltype(h.relative_index)>(t[0]), extract<decltype(h.to_sublattice)>(t[1]),
             extract<decltype(h.id)>(t[2]), extract<decltype(h.is_conjugate)>(t[3])};
    })
    ;

    class_<Sublattice>{"Sublattice"}
    .add_property("offset", return_copy(&Sublattice::offset), "Relative to global lattice offset")
    .def_readonly("onsite", &Sublattice::onsite, "Onsite energy")
    .def_readonly("alias", &Sublattice::alias,
                  "For supercells only: allows two sublattices to have the same ID")
    .add_property("hoppings", &Sublattice::hoppings, "List of :class:`~_pybinding.Hopping`")
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
    .add_property("ids", return_internal_copy([](SubIdRef const& s) { return arrayref(s.ids); }))
    .add_property("name_map", return_copy([](SubIdRef const& s) { return s.name_map; }))
    ;

    class_<Lattice>{"Lattice",
                    init<Cartesian, optional<Cartesian, Cartesian>>(args("a1", "a2", "a3"))}
    .def("_add_sublattice", &Lattice::add_sublattice,
         args("self", "name", "offset", "onsite_potential", "alias"))
    .def("_add_hopping", &Lattice::add_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "energy"))
    .def("_register_hopping_energy", &Lattice::register_hopping_energy, args("self", "energy"))
    .def("_add_registered_hopping", &Lattice::add_registered_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "id"))
    .add_property("vectors", &Lattice::vectors, "Primitive lattice vectors")
    .add_property("sublattices", &Lattice::sublattices, "List of :class:`~_pybinding.Sublattice`")
    .add_property("hopping_energies", &Lattice::hopping_energies,
                  "Unique energies indexed by hopping IDs")
    .add_property("sub_name_map", &Lattice::sub_name_map, "")
    .add_property("hop_name_map", &Lattice::hop_name_map, "")
    .add_property("offset", return_copy(&Lattice::offset), &Lattice::set_offset, R"(
        Global lattice offset: sublattice offsets are defined relative to this

        It must be within half the length of a primitive lattice vector.)")
    .def_readwrite("min_neighbors", &Lattice::min_neighbors, R"(
        Minimum number of neighbours required at each lattice site

        When constructing a finite-sized system, lattice sites with less neighbors
        than this minimum will be considered as "dangling" and they will be removed.)")
    .enable_pickling()
    .def("__getinitargs__", [](Lattice const& l) { return l.vectors; })
    .def("__getstate__", [](Lattice const& l) {
        return make_tuple(l.sublattices, l.hopping_energies, l.sub_name_map, l.hop_name_map,
                          l.min_neighbors, l.offset);
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
        l.min_neighbors = extract<decltype(l.min_neighbors)>(t[4]);
        l.offset = (len(t) >= 6) ? extract<decltype(l.offset)>(t[5]) : Cartesian(0, 0, 0);
    })
    ;

    class_<Primitive>{
        "Primitive", "Shape of the primitive unit cell",
        init<int, int, int> {args("self", "a1"_kw=1, "a2"_kw=1, "a3"_kw=1)}
    };

    class_<PyShape, noncopyable>{
        "Shape", init<PyShape::Vertices const&, object>(args("self", "vertices", "contains"))
    }
    .add_property("vertices", return_copy(&PyShape::vertices))
    .add_property("lattice_offset", return_copy(&PyShape::lattice_offset),
                  &PyShape::lattice_offset)
    ;

    class_<Line, bases<Shape>, noncopyable>{
        "Line", init<Cartesian, Cartesian>(args("self", "a", "b"))
    };

    class_<Polygon, bases<Shape>, noncopyable>{
        "Polygon", init<Polygon::Vertices const&>(args("self", "vertices"))
    };

    class_<PyFreeformShape, bases<Shape>, noncopyable>{
        "FreeformShape", init<object, Cartesian, Cartesian>(
            args("self", "contains", "width", "center")
        )
    };

    class_<TranslationalSymmetry>{
        "TranslationalSymmetry",
        init<float, float, float>{args("self", "a1", "a2", "a3")}
    };

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
