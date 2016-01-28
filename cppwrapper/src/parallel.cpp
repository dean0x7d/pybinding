#include "support/thread.hpp"
#include "python_support.hpp"

#include <boost/python/def.hpp>
#include <boost/python/class.hpp>

using namespace boost::python;
using namespace tbm;


void export_parallel() {
    using tbm::DeferredBase;
    class_<DeferredBase, noncopyable>{"DeferredBase", no_init}
    .def("compute", &DeferredBase::compute)
    .add_property("report", &DeferredBase::report)
    .add_property("result", internal_ref(&DeferredBase::result_uref))
    ;

    using tbm::Deferred;
    class_<Deferred<ArrayXf>, bases<DeferredBase>>{"DeferredXf", no_init};


    def("parallel_for", [](object sequence, object produce, object retire,
                           std::size_t num_threads, std::size_t queue_size)
    {
        auto const size = len(sequence);
        GILRelease main_thread_gil_release;

        tbm::parallel_for(
            size, num_threads, queue_size,
            [&produce, &sequence](size_t id) {
                GILEnsure gil_lock;
                object var = sequence[id];
                return extract<std::shared_ptr<DeferredBase>>{produce(var)}();
            },
            [](std::shared_ptr<DeferredBase>& deferred) {
                // no GIL lock -> computations run in parallel
                // but no Python code may be called here
                deferred->compute();
            },
            [&retire](std::shared_ptr<DeferredBase> deferred, size_t id) {
                GILEnsure gil_lock;
                retire(deferred, id);
                deferred.reset();
            }
        );
    }, args("sequence", "produce", "retire", "num_threads", "queue_size"));
}
