#include "detail/thread.hpp"
#include "wrappers.hpp"
using namespace cpb;

void wrap_parallel(py::module& m) {
    py::class_<DeferredBase, std::shared_ptr<DeferredBase>>(m, "DeferredBase", py::dynamic_attr())
        .def("compute", &DeferredBase::compute)
        .def_property_readonly("report", &DeferredBase::report)
        .def_property_readonly("result", &DeferredBase::result_uref);

    using DeferredXd = Deferred<ArrayXd>;
    py::class_<DeferredXd, std::shared_ptr<DeferredXd>, DeferredBase>(m, "DeferredXd");

    m.def("parallel_for", [](py::object sequence, py::object produce, py::object retire,
                             std::size_t num_threads, std::size_t queue_size) {
        auto const size = len(sequence);
        py::gil_scoped_release gil_release;

        parallel_for(
            size, num_threads, queue_size,
            [&produce, &sequence](size_t id) {
                py::gil_scoped_acquire gil_acquire;
                py::object var = sequence[py::cast(id)];
                return produce(var).cast<std::shared_ptr<DeferredBase>>();
            },
            [](std::shared_ptr<DeferredBase>& deferred) {
                // no GIL lock -> computations run in parallel
                // but no Python code may be called here
                deferred->compute();
            },
            [&retire](std::shared_ptr<DeferredBase> deferred, size_t id) {
                py::gil_scoped_acquire gil_acquire;
                retire(deferred, id);
                deferred.reset();
            }
        );
    }, "sequence"_a, "produce"_a, "retire"_a, "num_threads"_a, "queue_size"_a);
}
