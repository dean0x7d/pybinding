#include "leads/Leads.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_leads(py::module& m) {
    py::class_<leads::Spec>(m, "LeadSpec")
        .def_readonly("axis", &leads::Spec::axis)
        .def_readonly("sign", &leads::Spec::sign)
        .def_readonly("shape", &leads::Spec::shape)
    ;

    py::class_<Lead>(m, "Lead")
        .def_property_readonly("spec", &Lead::spec)
        .def_property_readonly("indices", [](Lead const& l) { return arrayref(l.indices()); })
        .def_property_readonly("system", &Lead::system)
        .def_property_readonly("h0", [](Lead const& l) { return l.h0().csrref(); })
        .def_property_readonly("h1", [](Lead const& l) { return l.h1().csrref(); })
    ;

    py::class_<Leads>(m, "Leads")
        .def("__len__", &Leads::size)
        .def("__getitem__", &Leads::operator[])
    ;
}
