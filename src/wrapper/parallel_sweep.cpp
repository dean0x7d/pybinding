#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include "python_support.hpp"

#include "Model.hpp"
#include "support/thread.hpp"
using namespace boost::python;

using ModelPtr = std::shared_ptr<tbm::Model>;
using ResultPtr = std::shared_ptr<tbm::Result>;

struct Job {
    Job() = default;
    explicit Job(std::size_t id) : id(id) {}

    std::size_t id = 0;
    ModelPtr model;
    ResultPtr result;
};

void parallel_sweep(std::size_t size, std::size_t num_threads, std::size_t queue_size,
                    object make_model, object make_result, object report)
{
#ifdef TBM_USE_MKL
    MKLDisableThreading disable_mkl_internal_threading_if{num_threads > 1};
#endif
    auto&& work_queue = Queue<Job>{queue_size > 0 ? queue_size : num_threads};
    auto&& report_queue = Queue<Job>{};

    // Allow the new python threads to work
    GILRelease main_thread_gil_release;

    // [Python] This thread produces new jobs and adds them to the work queue
    std::thread production_thread([&] {
        QueueGuard<Job> guard(work_queue);
        for (std::size_t i = 0; i < size; ++i) {
            auto job = Job{i};
            gil_ensure([&] {
                job.model = extract<ModelPtr>{make_model(job.id)};
                job.result = extract<ResultPtr>{make_result(job.id)};
                // Hamiltonian construction relies on Python and must be done here
                job.model->hamiltonian();
            });
            work_queue.push(std::move(job));
        }
    });

    // [C++] Multiple threads consume the work queue and add the computed jobs to the report queue.
    // Python code must not be called here or really bad things will happen. Destroying the last
    // shared_ptr to an object created in Python (such as model or result) will also call Python.
    // This is why jobs must be moved, otherwise the last shared_ptr may be destroyed here.
    auto work_threads = std::vector<std::thread>{num_threads > 0 ? num_threads : 1};
    for (auto& thread : work_threads) {
        thread = std::thread([&] {
            QueueGuard<Job> guard{report_queue};
            while (auto maybe_job = work_queue.pop()) {
                auto job = maybe_job.get();
                job.model->calculate(*job.result);
                report_queue.push(std::move(job));
            }
        });
    }

    // [Python] This thread consumes the report queue
    std::thread report_thread([&] {
        while (auto maybe_job = report_queue.pop()) {
            GILEnsure gil_lock;
            auto job = maybe_job.get();
            report(job.model, job.result, job.id);
        }
    });

    production_thread.join();
    for (auto& thread : work_threads) {
        thread.join();
    }
    report_thread.join();
};

void export_parallel_sweep()
{
    def("parallel_sweep", parallel_sweep,
        args("size", "num_threads", "queue_size", "make_model", "make_result", "report")
    );
}
