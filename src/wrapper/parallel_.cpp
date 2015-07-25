#include "support/thread.hpp"
#include "python_support.hpp"

#include <boost/python/def.hpp>
#include <boost/python/class.hpp>

using namespace boost::python;


void export_parallel() {
    using tbm::DeferredBase;
    class_<DeferredBase, noncopyable>{"DeferredBase", no_init}
    .def("compute", &DeferredBase::compute)
    .def("report", &DeferredBase::report)
    .def("result", &DeferredBase::result_uref)
    ;

    using tbm::Deferred;
    class_<Deferred<ArrayXf>, bases<DeferredBase>>{"DeferredXf", no_init};


    def("sweep", [](size_t size, size_t num_threads, size_t queue_size,
                    object produce, object report)
    {
        GILRelease main_thread_gil_release;
        tbm::sweep(
            size, num_threads, queue_size,
            [&produce](size_t id) {
                GILEnsure gil_lock;
                return extract<std::shared_ptr<DeferredBase>>{produce(id)}();
            },
            [](std::shared_ptr<DeferredBase>& deferred) {
                // no GIL lock -> computations run in parallel
                // but no Python code may be called here
                deferred->compute();
            },
            [&report](std::shared_ptr<DeferredBase> deferred, size_t id) {
                GILEnsure gil_lock;
                report(deferred, id);
                deferred.reset();
            }
        );
    }, args("size", "num_threads", "queue_size", "produce", "report"));
}
