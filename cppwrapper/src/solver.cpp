#include "solver/Solver.hpp"
#include "solver/FEAST.hpp"

#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/make_constructor.hpp>

using namespace boost::python;

void export_solver() {
    using tbm::BaseSolver;

    class_<BaseSolver, noncopyable>{
        "Solver", "Abstract base solver", no_init
    }
    .def("solve", &BaseSolver::solve)
    .def("clear", &BaseSolver::clear)
    .def("report", &BaseSolver::report, args("self", "shortform"_kw=false))
    .def("calc_dos", &BaseSolver::calc_dos, args("self", "energies", "broadening"))
    .def("calc_spatial_ldos", &BaseSolver::calc_spatial_ldos, args("self", "energy", "broadening"))
    .add_property("model", internal_ref(&BaseSolver::get_model), &BaseSolver::set_model)
    .add_property("system", &BaseSolver::system)
    .add_property("eigenvalues", internal_ref(&BaseSolver::eigenvalues))
    .add_property("eigenvectors", internal_ref(&BaseSolver::eigenvectors))
    ;

#ifdef TBM_USE_FEAST
    using tbm::Solver;
    using tbm::FEAST;
    auto const feast_defaults = tbm::FEASTConfig{};
    class_<Solver<FEAST>, bases<BaseSolver>, noncopyable>{"FEAST", no_init}
    .def("__init__", make_constructor([](tbm::Model const& model, std::pair<double, double> energy,
                                         int size_guess, bool recycle, bool verbose) {
            tbm::FEASTConfig config;
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
#endif // TBM_USE_FEAST
}
