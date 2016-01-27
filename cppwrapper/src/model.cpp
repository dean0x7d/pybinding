#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "python_support.hpp"
#include <boost/python/class.hpp>

using namespace boost::python;

void export_core() {
    using tbm::Model;
    class_<Model>{"Model", init<tbm::Lattice const&>(args("self", "lattice"))}
    .def("add", &Model::set_primitive)
    .def("add", &Model::set_shape)
    .def("add", &Model::set_symmetry)
    .def("add", &Model::add_site_state_modifier)
    .def("add", &Model::add_position_modifier)
    .def("add", &Model::add_onsite_modifier)
    .def("add", &Model::add_hopping_modifier)
    .def("attach_lead", &Model::attach_lead)
    .def("set_wave_vector", &Model::set_wave_vector, args("self", "wave_vector"))
    .add_property("state_modifiers", &Model::state_modifiers)
    .add_property("position_modifiers", &Model::position_modifiers)
    .add_property("onsite_modifiers", &Model::onsite_modifiers)
    .add_property("hopping_modifiers", &Model::hopping_modifiers)
    .add_property("system", copy_value(&Model::system))
    .add_property("hamiltonian", copy_value(&Model::hamiltonian))
    .def("report", &Model::report,
         "Report of the last build operation: system and Hamiltonian")
    .def("clear_system_modifiers", &Model::clear_system_modifiers)
    .def("clear_hamiltonian_modifiers", &Model::clear_hamiltonian_modifiers)
    .def("clear_all_modifiers", &Model::clear_all_modifiers)
    ;
}
