#include "kpm/default/Compute.hpp"
#include "kpm/default/collectors.hpp"

#include "compute/kernel_polynomial.hpp"
#include "kpm/calc_moments.hpp"

#include "detail/thread.hpp"

namespace cpb { namespace kpm {

namespace {

template<class Matrix>
struct SelectAlgorithm {
    using scalar_t = typename Matrix::Scalar;

    Matrix const& h2;
    Starter const& starter;
    AlgorithmConfig const& config;
    OptimizedHamiltonian const& oh;
    DefaultCompute const& compute;

    template<template<class> class C, class Vector = typename C<scalar_t>::Vector>
    idx_t with(C<scalar_t>& collect) const {
        simd::scope_disable_denormals guard;

        starter.lock();
        auto const idx = starter.count;
        auto r0 = make_r0(starter, var::tag<Vector>{}, simd::traits<scalar_t>::size);
        starter.unlock();

        auto r1 = make_r1(h2, r0);
        collect.initial(r0, r1);

        if (config.interleaved) {
            calc_moments::interleaved(collect, std::move(r0), std::move(r1),
                                      h2, oh.map(), config.optimal_size);
        } else {
            calc_moments::basic(collect, std::move(r0), std::move(r1),
                                h2, oh.map(), config.optimal_size);
        }

        return idx;
    }

    void operator()(DiagonalMoments* m) {
        auto collect = DiagonalCollector<scalar_t>(m->num_moments);
        with<DiagonalCollector>(collect);
        m->data = std::move(collect.moments);
    }

    void operator()(BatchDiagonalMoments* m) {
        constexpr auto batch_size = static_cast<idx_t>(simd::traits<scalar_t>::size);
        auto const num_threads = compute.get_num_threads();

        auto num_batches = m->num_vectors / batch_size;
        auto num_singles = m->num_vectors % batch_size;

        // Heuristic: prefer SIMD execution when there's a low number of threads
        if (num_singles > num_threads * batch_size / 2) {
            num_batches += 1;
            num_singles = 0;
        }

        ThreadPool pool(num_threads);
        compute.progress_start(m->num_vectors);

        for (auto i = 0; i < num_batches; ++i) {
            pool.add([&]() {
                auto collect = BatchDiagonalCollector<scalar_t>(m->num_moments, batch_size);
                auto const idx = with<BatchDiagonalCollector>(collect);
                m->add(collect.moments, idx);
                compute.progress_update(batch_size, m->num_vectors);
            });
        }

        for (auto i = 0; i < num_singles; ++i) {
            pool.add([&]() {
                auto collect = DiagonalCollector<scalar_t>(m->num_moments);
                auto const idx = with<DiagonalCollector>(collect);
                m->add(collect.moments, idx);
                compute.progress_update(1, m->num_vectors);
            });
        }

        pool.join();
        compute.progress_finish(m->num_vectors);
    }

    void operator()(GenericMoments* m) {
        auto collect = GenericCollector<scalar_t>(m->num_moments, oh, m->alpha, m->beta, m->op);
        with<OffDiagonalCollector>(collect);
        m->data = std::move(collect.moments);
    }

    void operator()(MultiUnitMoments* m) {
        auto collect = MultiUnitCollector<scalar_t>(m->num_moments, m->idx);
        with<OffDiagonalCollector>(collect);
        m->data = std::move(collect.moments);
    }

    void operator()(DenseMatrixMoments* m) {
        auto collect = DenseMatrixCollector<scalar_t>(m->num_moments, oh, m->op);
        with<OffDiagonalCollector>(collect);
        m->data = std::move(collect.moments);
    }
};

struct SelectMatrix {
    MomentsRef m;
    Starter const& s;
    AlgorithmConfig const& ac;
    OptimizedHamiltonian const& oh;
    DefaultCompute const& compute;

    template<class Matrix>
    void operator()(Matrix const& h2) {
        var::apply_visitor(SelectAlgorithm<Matrix>{h2, s, ac, oh, compute}, m);
    }
};

} // anonymous namespace

DefaultCompute::DefaultCompute(idx_t num_threads, ProgressCallback progress_callback)
    : num_threads(num_threads > 0 ? num_threads : std::thread::hardware_concurrency()),
      progress_callback(progress_callback) {}

void DefaultCompute::moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                             OptimizedHamiltonian const& oh) const {
    var::apply_visitor(SelectMatrix{std::move(m), s, ac, oh, *this}, oh.matrix());
}

void DefaultCompute::progress_start(idx_t total) const {
    progress_update(-1, total);
}

void DefaultCompute::progress_update(idx_t delta, idx_t total) const {
    if (!progress_callback) { return; }
    progress_callback(delta, total);
}

void DefaultCompute::progress_finish(idx_t total) const {
    progress_update(total, total);
}

}} // namespace cpb::kpm
