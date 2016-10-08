#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "wrappers.hpp"

using namespace cpb;

void wrap_shape(py::module& m) {
    py::class_<Primitive>(m, "Primitive")
        .def(py::init<int, int, int>());

    py::class_<Shape>(m, "Shape")
        .def("contains", [](Shape const& s, ArrayXf x, ArrayXf y, ArrayXf z) {
            return s.contains(CartesianArray(x, y, z));
        })
        .def_readonly("vertices", &Shape::vertices)
        .def_readwrite("lattice_offset", &Shape::lattice_offset);

    py::class_<Line, Shape>(m, "Line")
        .def(py::init<Cartesian, Cartesian>());

    py::class_<Polygon, Shape>(m, "Polygon")
        .def(py::init<Polygon::Vertices const&>());

    py::class_<FreeformShape, Shape>(m, "FreeformShape")
         .def("__init__", [](FreeformShape& s, py::object f, Cartesian width, Cartesian center) {
             auto contains = [f](CartesianArray const& p) {
                 py::gil_scoped_acquire guard;
                 return f(arrayref(p.x), arrayref(p.y), arrayref(p.z)).cast<ArrayX<bool>>();
             };
             new (&s) FreeformShape(contains, width, center);
         });

    py::class_<TranslationalSymmetry>(m, "TranslationalSymmetry")
        .def(py::init<float, float, float>());
}
