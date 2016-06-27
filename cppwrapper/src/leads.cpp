#include "leads/Leads.hpp"

#include "python_support.hpp"
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace cpb;

void export_leads() {
    class_<leads::Spec>{"LeadSpec", no_init}
    .def_readonly("axis", &leads::Spec::axis)
    .def_readonly("sign", &leads::Spec::sign)
    .add_property("shape", return_reference(&leads::Spec::shape))
    ;

    class_<Lead>{"Lead", no_init}
    .add_property("spec", return_reference(&Lead::spec))
    .add_property("indices", return_arrayref(&Lead::indices))
    .add_property("system", return_reference(&Lead::system))
    .add_property("h0", return_internal_copy([](Lead const& m) { return m.h0().csrref(); }))
    .add_property("h1", return_internal_copy([](Lead const& m) { return m.h1().csrref(); }))
    ;

    class_<Leads, noncopyable>{"Leads", no_init}
    .def("__len__", &Leads::size)
    .def("__getitem__", &Leads::operator[])
    ;
}
