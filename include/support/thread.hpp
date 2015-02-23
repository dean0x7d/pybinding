#pragma once
#include <thread>
#include <queue>
#include <condition_variable>

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

#ifdef TBM_USE_MKL
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
