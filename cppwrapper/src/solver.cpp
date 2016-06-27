#include "solver/Solver.hpp"
#include "solver/FEAST.hpp"

#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/make_constructor.hpp>

using namespace boost::python;

void export_solver() {
    using cpb::BaseSolver;

    class_<BaseSolver, noncopyable>{
        "Solver", "Abstract base solver", no_init
    }
    .def("solve", &BaseSolver::solve)
    .def("clear", &BaseSolver::clear)
    .def("report", &BaseSolver::report, args("self", "shortform"_kw=false))
    .def("calc_dos", &BaseSolver::calc_dos, args("self", "energies", "broadening"))
    .def("calc_spatial_ldos", &BaseSolver::calc_spatial_ldos, args("self", "energy", "broadening"))
    .add_property("model", return_reference(&BaseSolver::get_model), &BaseSolver::set_model)
    .add_property("system", &BaseSolver::system)
    .add_property("eigenvalues", return_internal_copy(&BaseSolver::eigenvalues))
    .add_property("eigenvectors", return_internal_copy(&BaseSolver::eigenvectors))
    ;

#ifdef CPB_USE_FEAST
    using cpb::Solver;
    using cpb::FEAST;
    auto const feast_defaults = cpb::FEASTConfig{};
    class_<Solver<FEAST>, bases<BaseSolver>, noncopyable>{"FEAST", no_init}
    .def("__init__", make_constructor([](cpb::Model const& model, std::pair<double, double> energy,
                                         int size_guess, bool recycle, bool verbose) {
            cpb::FEASTConfig config;
            config.energy_min = static_cast<float>(energy.first);
            config.energy_max = static_cast<float>(energy.second);
            config.initial_size_guess = size_guess;
            config.recycle_subspace = recycle;
            config.is_verbose = verbose;

            return new Solver<FEAST>(model, config);
        },
        default_call_policies(),
        args("self", "model", "energy_range", "initial_size_guess",
             "recycle_subspace"_kw = feast_defaults.recycle_subspace,
             "is_verbose"_kw = feast_defaults.is_verbose)
        )
    )
    ;
#endif // CPB_USE_FEAST
}
