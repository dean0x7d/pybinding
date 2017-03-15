#include "Model.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_model(py::module& m) {
    py::class_<Model>(m, "Model")
        .def(py::init<Lattice const&>())
        .def("add", &Model::add | resolve<Primitive>())
        .def("add", &Model::add | resolve<Shape const&>())
        .def("add", &Model::add | resolve<TranslationalSymmetry const&>())
        .def("add", &Model::add | resolve<SiteStateModifier const&>())
        .def("add", &Model::add | resolve<PositionModifier const&>())
        .def("add", &Model::add | resolve<OnsiteModifier const&>())
        .def("add", &Model::add | resolve<HoppingModifier const&>())
        .def("add", &Model::add | resolve<HoppingGenerator const&>())
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
        .def("eval", &Model::eval)
        .def("report", &Model::report, "Return a string with information about the last build")
        .def_property_readonly("system_build_seconds", &Model::system_build_seconds)
        .def_property_readonly("hamiltonian_build_seconds", &Model::hamiltonian_build_seconds);

    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def_property_readonly("csrref", &Hamiltonian::csrref);
}
