#include "Model.hpp"

#include "python_support.hpp"
#include <boost/python/class.hpp>

using namespace boost::python;
using namespace cpb;

void export_core() {
    class_<Model>{"Model", init<Lattice const&>(args("self", "lattice"))}
    .def("add", &Model::set_primitive)
    .def("add", &Model::set_shape)
    .def("add", &Model::set_symmetry)
    .def("add", &Model::add_site_state_modifier)
    .def("add", &Model::add_position_modifier)
    .def("add", &Model::add_onsite_modifier)
    .def("add", &Model::add_hopping_modifier)
    .def("add", &Model::add_hopping_family)
    .def("attach_lead", &Model::attach_lead)
    .def("set_wave_vector", &Model::set_wave_vector, args("self", "k"), R"(
        Set the wave vector for periodic models
    
        Parameters
        ----------
        k : array_like
            Wave vector in reciprocal space.
    )")
    .add_property("state_modifiers", &Model::state_modifiers, "")
    .add_property("position_modifiers", &Model::position_modifiers, "")
    .add_property("onsite_modifiers", &Model::onsite_modifiers, "")
    .add_property("hopping_modifiers", &Model::hopping_modifiers, "")
    .add_property("system", return_copy(&Model::system))
    .add_property("raw_hamiltonian", return_copy(&Model::hamiltonian), "")
    .add_property("hamiltonian", return_internal_copy(
        [](Model const& m) { return m.hamiltonian().csrref(); })
    )
    .add_property("leads", return_reference(&Model::leads))
    .def("report", &Model::report, "Return a string with information about the last build")
    .def("_clear_system_modifiers", &Model::clear_system_modifiers)
    .def("_clear_hamiltonian_modifiers", &Model::clear_hamiltonian_modifiers)
    .def("_clear_all_modifiers", &Model::clear_all_modifiers)
    ;
}
