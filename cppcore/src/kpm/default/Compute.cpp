#include "kpm/default/Compute.hpp"
#include "kpm/default/collectors.hpp"

#include "compute/kernel_polynomial.hpp"
#include "kpm/calc_moments.hpp"

namespace cpb { namespace kpm {

namespace {

template<class Matrix>
struct SelectAlgorithm {
    using scalar_t = typename Matrix::Scalar;

    Matrix const& h2;
    Starter const& starter;
    AlgorithmConfig const& config;
    OptimizedHamiltonian const& oh;
    idx_t batch_size;

    template<template<class> class C, class Vector = typename C<scalar_t>::Vector>
    void with(C<scalar_t>& collect) const {
        using namespace calc_moments;
        simd::scope_disable_denormals guard;

        auto r0 = make_r0(starter, var::tag<Vector>{}, batch_size);
        auto r1 = make_r1(h2, r0);
        collect.initial(r0, r1);

        if (config.optimal_size && config.interleaved) {
            opt_size_and_interleaved(collect, std::move(r0), std::move(r1), h2, oh.map());
        } else if (config.interleaved) {
            interleaved(collect, std::move(r0), std::move(r1), h2, oh.map());
        } else if (config.optimal_size) {
            opt_size(collect, std::move(r0), std::move(r1), h2, oh.map());
        } else {
            basic(collect, std::move(r0), std::move(r1), h2);
        }
    }

    void operator()(DiagonalMoments* m) {
        auto collect = DiagonalCollector<scalar_t>(m->num_moments);
        with<DiagonalCollector>(collect);
        m->data = std::move(collect.moments);
    }

    void operator()(BatchDiagonalMoments* m) {
        if (m->num_vectors <= 2) {
            auto collect = DiagonalCollector<scalar_t>(m->num_moments);
            for (auto i = idx_t{0}; i < m->num_vectors; ++i) {
                with<DiagonalCollector>(collect);
                m->add(collect.moments);
            }
        } else {
            auto collect = BatchDiagonalCollector<scalar_t>(m->num_moments, batch_size);
            for (auto i = idx_t{0}; i < m->num_vectors; i += batch_size) {
                with<BatchDiagonalCollector>(collect);
                m->add(collect.moments);
            }
        }
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
    idx_t batch_size;

    template<class Matrix>
    void operator()(Matrix const& h2) {
        m.match(SelectAlgorithm<Matrix>{h2, s, ac, oh, batch_size});
    }
};

struct BatchSize {
    template<class scalar_t>
    idx_t operator()(var::tag<scalar_t>) const {
        return static_cast<idx_t>(simd::traits<scalar_t>::size);
    }
};

} // anonymous namespace

void DefaultCompute::moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                             OptimizedHamiltonian const& oh) const {
    auto const batch_size = var::apply_visitor(BatchSize{}, oh.scalar_tag());
    var::apply_visitor(SelectMatrix{std::move(m), s, ac, oh, batch_size}, oh.matrix());
}

}} // namespace cpb::kpm
