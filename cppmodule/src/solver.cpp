#include "solver/Solver.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_solver(py::module& m) {
    py::class_<BaseSolver>(m, "Solver");
}
