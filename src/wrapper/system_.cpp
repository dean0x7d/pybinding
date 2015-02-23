#include "system/System.hpp"
#include "system/Shape.hpp"
#include "system/Lattice.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"

#include <boost/python/class.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/pure_virtual.hpp>
#include "python_support.hpp"
#include "converters/tuple.hpp"
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
    auto rbv = return_value_policy<return_by_value>{};

    class_<System::Boundary>{"Boundary", no_init}
    .add_property("shift", make_getter(&System::Boundary::shift, rbv))
    .add_property("matrix", &System::Boundary::matrix_uref)
    ;
    to_python_converter<std::vector<System::Boundary>, vector_to_list<System::Boundary>>{};

    class_<System, std::shared_ptr<System>>{"System", no_init}
    .def("find_nearest", &System::find_nearest, args("self", "position", "sublattice"_a=-1),
         "Find the index of the atom closest to the given coordiantes.")
    .add_property("num_sites", &System::num_sites)
    .add_property("x", const_ref(&System::x))
    .add_property("y", const_ref(&System::y))
    .add_property("z", const_ref(&System::z))
    .add_property("sublattice", by_value(&System::sublattice))
    .add_property("boundaries", by_value(&System::boundaries))
    .add_property("_matrix", &System::matrix_uref)
    .def_readonly("report", &System::report)
    ;
    register_ptr_to_python<std::shared_ptr<const System>>();

    using tbm::Lattice;
    class_<Lattice, noncopyable>{
        "Lattice", init<int>{args("self", "min_neighbours"_a=1)}
    }
    .def("add_vector", &Lattice::add_vector, args("self", "primitive_vector"))
    .def("create_sublattice", &Lattice::create_sublattice,
         args("self", "offset", "onsite_potential"_a=.0f, "alias"_a=-1))
    .def("add_hopping", &Lattice::add_hopping,
         args("self", "relative_index", "from_sublattice", "to_sublattice", "hopping_energy"))
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
    .add_property("r", make_getter(&Circle::radius, rbv), make_setter(&Circle::radius))
    .add_property("center", make_getter(&Circle::_center, rbv), make_setter(&Circle::_center))
    ;
    
    using tbm::Polygon;
    class_<Polygon, bases<Shape>, noncopyable> {
        "Polygon", "Shape defined by a list of vertices", init<> {arg("self")}
    }
    .add_property("x", make_getter(&Polygon::x, rbv), make_setter(&Polygon::x))
    .add_property("y", make_getter(&Polygon::y, rbv), make_setter(&Polygon::y))
    .add_property("offset", make_getter(&Polygon::offset, rbv), make_setter(&Polygon::offset))
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
