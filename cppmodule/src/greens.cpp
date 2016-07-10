#include "greens/KPM.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_greens(py::module& m) {
    py::class_<BaseGreens>(m, "Greens");
}
