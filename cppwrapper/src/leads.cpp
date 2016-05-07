#include "leads/Leads.hpp"

#include "python_support.hpp"
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace tbm;

void export_leads() {
    class_<Lead>{"Lead", no_init}
    .add_property("shift", return_copy(&Lead::shift))
    .add_property("indices", return_arrayref(&Lead::indices))
    .add_property("outer_hoppings", return_internal_copy(&Lead::outer_hoppings))
    .add_property("inner_hoppings", return_internal_copy(&Lead::inner_hoppings))
    .add_property("h0", return_internal_copy([](Lead const& m) { return m.h0().csrref(); }))
    .add_property("h1", return_internal_copy([](Lead const& m) { return m.h1().csrref(); }))
    ;

    class_<Leads, noncopyable>{"Leads", no_init}
    .def("__len__", &Leads::size)
    .def("__getitem__", &Leads::operator[])
    ;
}
