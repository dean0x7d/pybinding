#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "cast.hpp"

namespace py = pybind11;
using namespace py::literals;

void wrap_greens(py::module& m);
void wrap_lattice(py::module& m);
void wrap_model(py::module& m);
void wrap_solver(py::module& m);
void wrap_system(py::module& m);
