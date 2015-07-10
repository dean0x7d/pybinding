#include "solver/FEAST.hpp"

#include <boost/python/class.hpp>
#include "python_support.hpp"
using namespace boost::python;

void export_solver() {
    using tbm::Solver;

    class_<Solver, noncopyable>{
        "Solver", "Abstract base solver", no_init
    }
    .def("set_model", &Solver::set_model, args("self", "model"))
    .def("solve", &Solver::solve)
    .def("report", &Solver::report, args("self", "shortform"_kw=false))
    .def("calc_dos", &Solver::calc_dos, args("self", "energies", "broadening"))
    .def("calc_ldos", &Solver::calc_ldos, args("self", "energy", "broadening", "sublattice"_kw=-1))
    .add_property("eigenvalues", internal_ref(&Solver::eigenvalues))
    .add_property("eigenvectors", internal_ref(&Solver::eigenvectors))
    .add_property("system", &Solver::system)
    ;

#ifdef TBM_USE_FEAST
    using tbm::FEAST;

    auto const feast_defaults = tbm::FEASTConfig{};
    class_<FEAST, bases<Solver>, noncopyable>{
        "FEAST", "FEAST eigensolver.",
        init<tbm::Model const&, std::pair<double, double>, int, bool, bool>{
            args("self", "model", "energy_range", "initial_size_guess",
                 "recycle_subspace"_kw=feast_defaults.recycle_subspace,
                 "is_verbose"_kw=feast_defaults.is_verbose)
        }
    }
    ;
#endif // TBM_USE_FEAST
}
