#include "Model.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_model(py::module& m) {
    py::class_<Model>(m, "Model")
        .def(py::init<Lattice const&>())
        .def("add", &Model::set_primitive)
        .def("add", &Model::set_shape)
        .def("add", &Model::set_symmetry)
        .def("add", &Model::add_site_state_modifier)
        .def("add", &Model::add_position_modifier)
        .def("add", &Model::add_onsite_modifier)
        .def("add", &Model::add_hopping_modifier)
        .def("add", &Model::add_hopping_family)
        .def("attach_lead", &Model::attach_lead)
        .def("set_wave_vector", &Model::set_wave_vector, "k"_a, R"(
            Set the wave vector for periodic models

            Parameters
            ----------
            k : array_like
                Wave vector in reciprocal space.
        )")
        .def_property_readonly("system", &Model::system)
        .def_property_readonly("raw_hamiltonian", &Model::hamiltonian)
        .def_property_readonly("hamiltonian", [](Model const& self) {
            return self.hamiltonian().csrref();
        })
        .def_property_readonly("leads", &Model::leads)
        .def("report", &Model::report, "Return a string with information about the last build");

    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def_property_readonly("csrref", &Hamiltonian::csrref);
}
