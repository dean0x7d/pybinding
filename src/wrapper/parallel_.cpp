#include "support/thread.hpp"
#include "python_support.hpp"

#include <boost/python/def.hpp>
#include <boost/python/class.hpp>

using namespace boost::python;


void export_parallel() {
    using tbm::DeferredBase;
    class_<DeferredBase, noncopyable>{"DeferredBase", no_init}
    .def("compute", &DeferredBase::compute)
    .add_property("report", &DeferredBase::report)
    .add_property("result", internal_ref(&DeferredBase::result_uref))
    ;

    using tbm::Deferred;
    class_<Deferred<ArrayXf>, bases<DeferredBase>>{"DeferredXf", no_init};


    def("sweep", [](object variables, object produce, object report,
                    size_t num_threads, size_t queue_size)
    {
        auto size = len(variables);
        GILRelease main_thread_gil_release;

        tbm::sweep(
            size, num_threads, queue_size,
            [&produce, &variables](size_t id) {
                GILEnsure gil_lock;
                object var = variables[id];
                return extract<std::shared_ptr<DeferredBase>>{produce(var)}();
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
    }, args("variables", "produce", "report", "num_threads", "queue_size"));
}
