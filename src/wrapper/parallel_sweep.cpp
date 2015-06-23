#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include "python_support.hpp"

#include "Model.hpp"
#include "Greens/KPM.hpp"
#include "support/thread.hpp"
using namespace boost::python;

class ComputeKernel {
public:
    virtual void compute() = 0;
    virtual std::string report() = 0;
};

class KPMldos : public ComputeKernel {
public:
    KPMldos(std::shared_ptr<tbm::KPM> const& kpm,
            ArrayXd energy, float broadening, Cartesian position, short sublattice = -1)
        : kpm(kpm), energy(energy), broadening(broadening),
          position(position), sublattice(sublattice) {}

    virtual void compute() final {
        ldos = kpm->calc_ldos(energy, broadening, position, sublattice);
    }

    virtual std::string report() final {
        return kpm->report(true);
    }

    std::shared_ptr<tbm::KPM> kpm;
    ArrayXd energy;
    float broadening;
    Cartesian position;
    short sublattice;
    ArrayXf ldos;
};

using ModelPtr = std::shared_ptr<tbm::Model>;
using ResultPtr = std::shared_ptr<ComputeKernel>;

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
    Queue<Job> work_queue{queue_size > 0 ? queue_size : num_threads};
    Queue<Job> report_queue{};

    // Allow the new python threads to work
    GILRelease main_thread_gil_release;

    // [Python] This thread produces new jobs and adds them to the work queue
    std::thread production_thread([&] {
        QueueGuard<Job> guard(work_queue);
        for (std::size_t i = 0; i < size; ++i) {
            auto job = Job{i};
            gil_ensure([&] {
                job.model = extract<ModelPtr>{make_model(job.id)};
                // Hamiltonian construction relies on Python and must be done here
                job.model->hamiltonian();
                job.result = extract<ResultPtr>{make_result(job.model, job.id)};
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
                job.result->compute();
                report_queue.push(std::move(job));
            }
        });
    }

    // [Python] This thread consumes the report queue
    std::thread report_thread([&] {
        while (auto maybe_job = report_queue.pop()) {
            GILEnsure gil_lock;
            auto job = maybe_job.get();
            report(job.result, job.id);
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

    class_<ComputeKernel, noncopyable>{"ComputeKernel", no_init}
    .def("compute", &ComputeKernel::compute)
    .def("report", &ComputeKernel::report)
    ;

    class_<KPMldos, std::shared_ptr<KPMldos>, bases<ComputeKernel>>{
        "KPMldos",
        init<const std::shared_ptr<tbm::KPM>&, ArrayXd, float, Cartesian, short>{
            args("self", "kpm", "energy", "broadening", "position", "sublattice"_kw=-1)
        }
    }
    .add_property("ldos", copy_value(&KPMldos::ldos))
    .add_property("energy", copy_value(&KPMldos::energy))
    ;
}
