#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"
#include <boost/python/class.hpp>
using namespace boost::python;
using tbm::Model;

void export_core()
{
    class_<Model>{"Model", "The main tight-binding interface object."}
    .def("add", &Model::set_lattice)
    .def("add", &Model::set_shape)
    .def("add", &Model::set_symmetry)
    .def("add", &Model::add_site_state_modifier)
    .def("add", &Model::add_position_modifier)
    .def("add", &Model::add_onsite_modifier)
    .def("add", &Model::add_hopping_modifier)
    .def("set_wave_vector", &Model::set_wave_vector, args("self", "wave_vector"))
    .add_property("lattice", &Model::lattice, &Model::set_lattice)
    .add_property("shape", &Model::shape, &Model::set_shape)
    .add_property("symmetry", &Model::symmetry, &Model::set_symmetry)
    .add_property("system", &Model::system)
    .add_property("hamiltonian", &Model::hamiltonian)
    .def("report", &Model::report,
         "Report of the last build operation: system and Hamiltonian")
    .def("clear_symmetry", &Model::clear_symmetry)
    .def("clear_system_modifiers", &Model::clear_system_modifiers)
    .def("clear_hamiltonian_modifiers", &Model::clear_hamiltonian_modifiers)
    .def("clear_all_modifiers", &Model::clear_all_modifiers)
    ;
}
