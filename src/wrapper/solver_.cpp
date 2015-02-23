#include "solver/FEAST.hpp"

#include <boost/python/class.hpp>
using namespace boost::python;

void export_solver()
{
    using tbm::Solver;
    using tbm::SolverFactory;

    class_<Solver, std::shared_ptr<Solver>, noncopyable> {
        "Solver", "Abstract base solver", no_init
    }
    .def("solve", &Solver::solve, arg("self"))
    .add_property("energy", &Solver::eigenvalues)
    .add_property("psi", &Solver::eigenvectors)
    ;

    class_<SolverFactory, noncopyable> {
        "SolverFactory", "Abstract base solver factory", no_init
    };

#ifdef TBM_USE_FEAST
    using tbm::FEASTFactory;
    using tbm::FEAST;

    FEAST<double>::Params defParams;
    class_<FEASTFactory, bases<SolverFactory>, noncopyable> {
        "FEAST", "FEAST solver for a finite system.",
        init<float, float, int, bool, bool> {
            (arg("self"), "min_energy", "max_energy", "size",
                arg("recycle") = FEASTFactory::defaults::recycle_subspace,
                arg("verbose") = FEASTFactory::defaults::is_verbose)
        }
    }
    .def("advanced", &FEASTFactory::advanced, return_value_policy<reference_existing_object>(),
         (arg("self"),
             arg("contour_points") = defParams.contour_points,
             arg("max_refinement_loops") = defParams.max_refinement_loops,
             arg("sp_stop_criteria") = defParams.sp_stop_criteria,
             arg("dp_stop_criteria") = defParams.dp_stop_criteria,
             arg("residual_stop_criteria") = defParams.residual_convergence
         )
    );

    /*class_<FEAST::Info> (
        "FEASTinfo", init<>(arg("self"))
    )
    .def_readonly("final_size", &FEAST::Info::final_size)
    .def_readonly("suggested_guess_size", &FEAST::Info::suggested_size)
    .def_readonly("refinement_loops", &FEAST::Info::refinement_loops)
    .def_readonly("recycle_warning", &FEAST::Info::recycle_warning)
    .def_readonly("error_trace", &FEAST::Info::error_trace)
    .def_readonly("max_residual", &FEAST::Info::max_residual)
    .def_readonly("return_code", &FEAST::Info::return_code)
    ;*/
#endif // TBM_USE_FEAST
}
