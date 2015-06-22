#include "system/System.hpp"
#include "system/Shape.hpp"
#include "system/Lattice.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/pure_virtual.hpp>
#include "python_support.hpp"
using namespace boost::python;


class PySiteStateModifier : public tbm::SiteStateModifier, public wrapper<tbm::SiteStateModifier> {
public:
    virtual void apply(ArrayX<bool>& is_valid, const CartesianArray& p) const override {
        object o = get_override("apply")(is_valid, p.x, p.y, p.z);
        is_valid = extract<ArrayX<bool>>(o);
    }
    void apply_dummy(ArrayX<bool>&, const ArrayXf&, const ArrayXf&, const ArrayXf&) const {}
};

class PyPositionModifier : public tbm::PositionModifier, public wrapper<tbm::PositionModifier> {
public:
    virtual void apply(CartesianArray& p) const override {
        tuple o = get_override("apply")(p.x, p.y, p.z);
        p.x = extract<ArrayXf>(o[0]);
        p.y = extract<ArrayXf>(o[1]);
        p.z = extract<ArrayXf>(o[2]);
    }
    void apply_dummy(ArrayXf&, ArrayXf&, ArrayXf&) const {}
};

void export_system()
{
    using tbm::System;

    class_<System::Boundary>{"Boundary", no_init}
    .add_property("shift", copy_value(&System::Boundary::shift))
    .add_property("matrix", sparse_uref(&System::Boundary::matrix))
    ;

    class_<System>{"System", no_init}
    .def("find_nearest", &System::find_nearest, args("self", "position", "sublattice"_kw=-1),
         "Find the index of the atom closest to the given coordiantes.")
    .add_property("num_sites", &System::num_sites)
    .add_property("x", dense_uref(&System::x))
    .add_property("y", dense_uref(&System::y))
    .add_property("z", dense_uref(&System::z))
    .add_property("sublattice", dense_uref(&System::sublattice))
    .add_property("boundaries", &System::boundaries)
    .add_property("matrix", sparse_uref(&System::matrix))
    .def_readonly("report", &System::report)
    ;

    using tbm::Hopping;
    class_<Hopping>{"Hopping", no_init}
    .add_property("relative_index", copy_value(&Hopping::relative_index))
    .def_readonly("to_sublattice", &Hopping::to_sublattice)
    .def_readonly("energy", &Hopping::energy)
    ;

    using tbm::Sublattice;
    class_<Sublattice>{"Sublattice", no_init}
    .add_property("offset", copy_value(&Sublattice::offset))
    .def_readonly("onsite", &Sublattice::onsite)
    .def_readonly("alias", &Sublattice::alias)
    .add_property("hoppings", &Sublattice::hoppings)
    ;

    using tbm::Lattice;
    class_<Lattice, noncopyable>{
        "Lattice", init<int>{args("self", "min_neighbours"_kw=1)}
    }
    .def("add_vector", &Lattice::add_vector, args("self", "primitive_vector"))
    .def("create_sublattice", &Lattice::create_sublattice,
         args("self", "offset", "onsite_potential"_kw=.0f, "alias"_kw=-1))
    .def("add_hopping", &Lattice::add_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "hopping_energy"))
    .add_property("vectors", &Lattice::vectors)
    .add_property("sublattices", &Lattice::sublattices)
    .def_readwrite("min_neighbors", &Lattice::min_neighbours)
    ;
    
    using tbm::Shape;
    class_<Shape, noncopyable> {"Shape", no_init};
    
    using tbm::Primitive;
    class_<Primitive, bases<Shape>, noncopyable> {
        "Primitive", "Shape of the primitive unit cell",
        init<Cartesian, bool> {(arg("self"), "length", arg("nanometers")=false)}
    };

    using tbm::Circle;
    class_<Circle, bases<Shape>, noncopyable> {
        "Circle", "Perfect circle",
        init<float, optional<Cartesian>> {
            (arg("self"), "radius", "center")
        }
    }
    .add_property("r", &Circle::radius, &Circle::radius)
    .add_property("center", &Circle::_center, &Circle::_center)
    ;
    
    using tbm::Polygon;
    class_<Polygon, bases<Shape>, noncopyable> {
        "Polygon", "Shape defined by a list of vertices", init<> {arg("self")}
    }
    .add_property("x", copy_value(&Polygon::x), &Polygon::x)
    .add_property("y", copy_value(&Polygon::y), &Polygon::y)
    .add_property("offset", copy_value(&Polygon::offset), &Polygon::offset)
    ;

    using tbm::Symmetry;
    class_<Symmetry, noncopyable> {"Symmetry", no_init};

    using tbm::Translational;
    class_<Translational, bases<Symmetry>, noncopyable> {
        "Translational", "Periodic boundary condition.", 
        init<Cartesian> {(arg("self"), "length")}
    };

    class_<PySiteStateModifier, noncopyable>{"SiteStateModifier"}
    .def("apply", pure_virtual(&PySiteStateModifier::apply_dummy), args("self", "site_state", "x", "y", "z"))
    ;
    class_<PyPositionModifier, noncopyable>{"PositionModifier"}
    .def("apply", pure_virtual(&PyPositionModifier::apply_dummy), args("self", "x", "y", "z"))
    ;
}
