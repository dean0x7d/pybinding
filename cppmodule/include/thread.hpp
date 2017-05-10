#pragma once
#include <pybind11/pybind11.h>
#include "detail/thread.hpp"

namespace py = pybind11;

namespace cpb {

class DeferredBase {
public:
    virtual ~DeferredBase() = default;

    virtual py::object solver() const = 0;
    virtual void compute() = 0;
    virtual py::object result() = 0;
};

template<class Result>
class Deferred : public DeferredBase {
public:
    Deferred(py::object solver, std::function<Result()> compute)
        : _solver(std::move(solver)), _compute(std::move(compute)) {}

    py::object solver() const final { return _solver; }

    void compute() final {
        if (is_computed) { return; }

        _result = _compute();
        is_computed = true;
    }

    py::object result() final { compute(); return py::cast(_result); }

private:
    py::object _solver;
    std::function<Result()> _compute;
    Result _result;

    bool is_computed = false;
};

} // namespace cpb
