#pragma once
#include "detail/config.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "cast.hpp"
#include "resolve.hpp"

namespace py = pybind11;
using namespace py::literals;

using release_gil = py::call_guard<py::gil_scoped_release>;

void wrap_greens(py::module& m);
void wrap_lattice(py::module& m);
void wrap_leads(py::module& m);
void wrap_model(py::module& m);
void wrap_modifiers(py::module& m);
void wrap_parallel(py::module& m);
void wrap_shape(py::module& m);
void wrap_solver(py::module& m);
void wrap_system(py::module& m);

void wrapper_tests(py::module& m);
