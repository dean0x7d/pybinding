#pragma once
#include "utils/Chrono.hpp"
#include "numeric/dense.hpp"

#include <thread>
#include <queue>
#include <condition_variable>


namespace cpb { namespace detail {

template<class T>
class Queue {
public:
    struct Maybe {
        Maybe() = default;
        Maybe(T&& value) : value(std::move(value)), is_valid(true) {}

        operator bool() { return is_valid; }
        T get() { return std::move(value); }

    private:
        T value;
        bool is_valid = false;
    };

public:
    Queue() = default;
    Queue(std::size_t max_size) : max_size(max_size) {}
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    void add_producer() {
        std::unique_lock<std::mutex> lk(m);
        num_producers++;
        if (num_producers > 0)
            is_closed = false;
    }

    void remove_producer() {
        std::unique_lock<std::mutex> lk(m);
        num_producers--;
        if (num_producers <= 0)
            is_closed = true;
        lk.unlock();
        consumption_cv.notify_all();
    }

    Maybe pop() {
        std::unique_lock<std::mutex> lk(m);
        consumption_cv.wait(lk, [&] { return !q.empty() || is_closed; });

        if (q.empty())
            return {};

        auto val = q.front();
        q.pop();
        lk.unlock();
        production_cv.notify_one();
        return std::move(val);
    }

    void push(const T& item) {
        std::unique_lock<std::mutex> lk(m);
        production_cv.wait(lk, [&] { return q.size() < max_size; });
        q.push(item);
        lk.unlock();
        consumption_cv.notify_one();
    }

    void push(T&& item) {
        std::unique_lock<std::mutex> lk(m);
        production_cv.wait(lk, [&] { return q.size() < max_size; });
        q.push(std::move(item));
        lk.unlock();
        consumption_cv.notify_one();
    }

    std::size_t size() const { return q.size(); }

private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable production_cv;
    std::condition_variable consumption_cv;

    bool is_closed = false;
    int num_producers = 0;
    std::size_t max_size = std::numeric_limits<std::size_t>::max();
};

template<class T>
class QueueGuard {
    Queue<T>& wq;
public:
    QueueGuard(Queue<T>& q) : wq(q) { wq.add_producer(); }
    ~QueueGuard() { wq.remove_producer(); }
};

#ifdef CPB_USE_MKL
# include <mkl.h>

class MKLDisableThreading {
public:
    MKLDisableThreading(bool condition) : num_threads{mkl_get_max_threads()} {
        if (condition)
            mkl_set_num_threads(1);
    }
    ~MKLDisableThreading() { mkl_set_num_threads(num_threads); }

private:
    int num_threads;
};

#endif

} // namespace detail

class DeferredBase {
public:
    virtual void compute() = 0;
    virtual std::string report() = 0;
    virtual ArrayConstRef result_uref() = 0;
};

template<class Result>
class Deferred : public DeferredBase {
    using Compute = std::function<void(Result&)>;
    using Report = std::function<std::string()>;

public:
    Deferred(Compute compute, Report report)
        : _compute(compute), _report(report)
    {}

    void compute() final {
        if (is_computed)
            return;

        timer.tic();
        _compute(_result);
        timer.toc();
        is_computed = true;
    }

    std::string report() final {
        return _report() + " " + timer.str();
    }

    Result result() {
        compute();
        return _result;
    }

    ArrayConstRef result_uref() final {
        compute();
        return arrayref(std17::as_const(_result));
    }

private:
    Compute _compute;
    Report _report;
    Result _result;

    bool is_computed = false;
    Chrono timer;
};


template<class Produce, class Compute, class Retire>
void parallel_for(size_t size, size_t num_threads, size_t queue_size,
                  Produce produce, Compute compute, Retire retire) {
#ifdef CPB_USE_MKL
    detail::MKLDisableThreading disable_mkl_internal_threading_if{num_threads > 1};
#endif

    using Value = decltype(produce(size_t{}));
    struct Job {
        size_t id;
        Value value;
    };

    detail::Queue<Job> work_queue{queue_size > 0 ? queue_size : num_threads};
    detail::Queue<Job> retirement_queue{};

    // This thread produces new jobs and adds them to the work queue
    std::thread production_thread([&] {
        detail::QueueGuard<Job> guard{work_queue};
        for (auto id = size_t{0}; id < size; ++id) {
            work_queue.push({id, produce(id)});
        }
    });

    // Multiple compute threads consume the work queue
    // and send the completed jobs to the retirement queue
    auto work_threads = std::vector<std::thread>{num_threads};
    for (auto& thread : work_threads) {
        thread = std::thread([&] {
            detail::QueueGuard<Job> guard{retirement_queue};
            while (auto maybe_job = work_queue.pop()) {
                auto job = maybe_job.get();
                compute(job.value);
                retirement_queue.push(std::move(job));
            }
        });
    }

    // This thread consumes the retirement queue
    std::thread report_thread([&] {
        while (auto maybe_job = retirement_queue.pop()) {
            auto job = maybe_job.get();
            retire(std::move(job.value), job.id);
        }
    });

    production_thread.join();
    for (auto& thread : work_threads) {
        thread.join();
    }
    report_thread.join();
}

} // namespace cpb

