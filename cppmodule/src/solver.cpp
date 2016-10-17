#include "solver/Solver.hpp"
#include "solver/FEAST.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_solver(py::module& m) {
    py::class_<BaseSolver>(m, "Solver")
        .def("solve", &BaseSolver::solve)
        .def("clear", &BaseSolver::clear)
        .def("report", &BaseSolver::report, "shortform"_a=false)
        .def("calc_dos", &BaseSolver::calc_dos, "energies"_a, "broadening"_a)
        .def("calc_spatial_ldos", &BaseSolver::calc_spatial_ldos, "energy"_a, "broadening"_a)
        .def_property("model", &BaseSolver::get_model, &BaseSolver::set_model)
        .def_property_readonly("system", &BaseSolver::system)
        .def_property_readonly("eigenvalues", &BaseSolver::eigenvalues)
        .def_property_readonly("eigenvectors", &BaseSolver::eigenvectors);

#ifdef CPB_USE_FEAST
    auto const feast_defaults = FEASTConfig();
    py::class_<Solver<FEAST>, BaseSolver>(m, "FEAST")
        .def("__init__", [](Solver<FEAST>& self, Model const& model,
                            std::pair<float, float> energy, int size_guess,
                            bool recycle, bool verbose) {
                 FEASTConfig config;
                 config.energy_min = energy.first;
                 config.energy_max = energy.second;
                 config.initial_size_guess = size_guess;
                 config.recycle_subspace = recycle;
                 config.is_verbose = verbose;

                 new (&self) Solver<FEAST>(model, config);

             },
             "model"_a, "energy_range"_a, "initial_size_guess"_a,
             "recycle_subspace"_a=feast_defaults.recycle_subspace,
             "is_verbose"_a=feast_defaults.is_verbose
        );
#endif // CPB_USE_FEAST
}
