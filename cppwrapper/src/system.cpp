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
using namespace tbm;


class PySiteStateModifier : public tbm::SiteStateModifierImpl,
                            public wrapper<tbm::SiteStateModifierImpl> {
public:
    virtual void apply(ArrayX<bool>& is_valid, CartesianArray const& p,
                       ArrayX<tbm::sub_id> const& s) const override {
        object result = get_override("apply")(
            arrayref(is_valid),
            arrayref(p.x), arrayref(p.y), arrayref(p.z),
            arrayref(s)
        );
        extract_array(is_valid, result);
    }
};

class PyPositionModifier : public tbm::PositionModifierImpl,
                           public wrapper<tbm::PositionModifierImpl> {
public:
    virtual void apply(CartesianArray& p, ArrayX<tbm::sub_id> const& s) const override {
        tuple result = get_override("apply")(
            arrayref(p.x), arrayref(p.y), arrayref(p.z), arrayref(s)
        );
        extract_array(p.x, result[0]);
        extract_array(p.y, result[1]);
        extract_array(p.z, result[2]);
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

void export_system() {
    using Boundary = tbm::System::Boundary;
    class_<Boundary>{"Boundary", no_init}
    .add_property("shift", copy_value(&Boundary::shift))
    .add_property("hoppings", sparse_uref(&Boundary::hoppings))
    ;

    using Port = tbm::System::Port;
    class_<Port>{"Port", no_init}
    .add_property("shift", copy_value(&Port::shift))
    .add_property("indices", copy_value(&Port::indices))
    .add_property("outer_hoppings", sparse_uref(&Port::outer_hoppings))
    .add_property("inner_hoppings", sparse_uref(&Port::inner_hoppings))
    ;

    using tbm::System;
    class_<System, std::shared_ptr<System>, noncopyable>{"System", no_init}
    .def("find_nearest", &System::find_nearest, args("self", "position", "sublattice"_kw=-1),
         "Find the index of the atom closest to the given coordiantes.")
    .add_property("num_sites", &System::num_sites)
    .add_property("positions", internal_ref(&System::positions))
    .add_property("sublattices", dense_uref(&System::sublattices))
    .add_property("hoppings", sparse_uref(&System::hoppings))
    .add_property("boundaries", &System::boundaries)
    .add_property("ports", &System::ports)
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

    using tbm::Lattice;
    class_<Lattice>{"Lattice",
                    init<Cartesian, optional<Cartesian, Cartesian>>(args("a1", "a2", "a3"))}
    .def("_add_sublattice", &Lattice::add_sublattice,
         args("self", "offset", "onsite_potential"_kw=.0f, "alias"_kw=-1))
    .def("_add_hopping", &Lattice::add_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "energy"))
    .def("_register_hopping_energy", &Lattice::register_hopping_energy, args("self", "energy"))
    .def("_add_registered_hopping", &Lattice::add_registered_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "id"))
    .add_property("vectors", &Lattice::vectors, &Lattice::vectors)
    .add_property("sublattices", &Lattice::sublattices, [](Lattice& l, std::vector<Sublattice> s) {
        l.has_onsite_energy = std::any_of(s.begin(), s.end(), [](Sublattice const& sub) {
            return sub.onsite != 0;
        });
        l.sublattices = std::move(s);
    })
    .add_property("hopping_energies", &Lattice::hopping_energies,
                  [](Lattice& l, std::vector<std::complex<double>> h) {
        l.has_complex_hopping = std::any_of(h.begin(), h.end(), [](std::complex<double> energy) {
            return energy.imag() != .0;
        });
        l.hopping_energies = std::move(h);
    })
    .def_readwrite("min_neighbors", &Lattice::min_neighbours)
    .enable_pickling()
    .def("__getinitargs__", [](Lattice const& l) { return l.vectors; })
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

    class_<PySiteStateModifier, noncopyable>{"SiteStateModifier"};
    class_<PyPositionModifier, noncopyable>{"PositionModifier"};
}
