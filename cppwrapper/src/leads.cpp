#include "leads/Leads.hpp"

#include "python_support.hpp"
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace tbm;

void export_leads() {
    class_<Lead>{"Lead", no_init}
    .add_property("shift", copy_value(&Lead::shift))
    .add_property("indices", copy_value(&Lead::indices))
    .add_property("outer_hoppings", internal_ref(&Lead::outer_hoppings))
    .add_property("inner_hoppings", internal_ref(&Lead::inner_hoppings))
    .add_property("h0", copy_value(&Lead::h0))
    .add_property("h1", copy_value(&Lead::h1))
    ;

    class_<Leads>{"Leads", no_init}
    .def("__len__", &Leads::size)
    .def("__getitem__", &Leads::operator[])
    ;
}
