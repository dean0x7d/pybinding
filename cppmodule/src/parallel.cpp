#include "wrappers.hpp"
#include "thread.hpp"
using namespace cpb;

void wrap_parallel(py::module& m) {
    py::class_<DeferredBase, std::shared_ptr<DeferredBase>>(m, "DeferredBase", py::dynamic_attr())
        .def("compute", &DeferredBase::compute)
        .def_property_readonly("solver", &DeferredBase::solver)
        .def_property_readonly("result", &DeferredBase::result);

    using DeferredXd = Deferred<Eigen::ArrayXd>;
    py::class_<DeferredXd, std::shared_ptr<DeferredXd>, DeferredBase>(m, "DeferredXd");

    m.def("parallel_for", [](py::object sequence, py::object produce, py::object retire,
                             std::size_t num_threads, std::size_t queue_size) {
        auto const size = py::len(sequence);
        py::gil_scoped_release gil_release;

        struct Job {
            py::object py;
            std::shared_ptr<DeferredBase> cpp;
        };

        parallel_for(
            size, num_threads, queue_size,
            [&produce, &sequence](size_t id) {
                py::gil_scoped_acquire gil_acquire;
                py::object var = sequence[py::cast(id)];
                py::object obj = produce(var);
                return Job{obj, obj.cast<std::shared_ptr<DeferredBase>>()};
            },
            [](Job& job) {
                // no GIL lock -> computations run in parallel
                // but no Python code may be called here
                job.cpp->compute();
            },
            [&retire](Job job, size_t id) {
                py::gil_scoped_acquire gil_acquire;
                retire(job.py, id);
                job.py.release();
            }
        );
    }, "sequence"_a, "produce"_a, "retire"_a, "num_threads"_a, "queue_size"_a);
}
