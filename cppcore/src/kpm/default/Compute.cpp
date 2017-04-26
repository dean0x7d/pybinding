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

    template<template<class> class C, class Vector = typename C<scalar_t>::Vector>
    void with(C<scalar_t>& collect) const {
        using namespace calc_moments;
        simd::scope_disable_denormals guard;

        auto r0 = make_r0(starter, var::tag<Vector>{});
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
        auto collect = BatchDiagonalCollector<scalar_t>(m->num_moments, m->batch_size);
        with<BatchDiagonalCollector>(collect);
        m->data = std::move(collect.moments);
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

    template<class Matrix>
    void operator()(Matrix const& h2) {
        m.match(SelectAlgorithm<Matrix>{h2, s, ac, oh});
    }
};

struct BatchSize {
    template<class scalar_t>
    idx_t operator()(var::tag<scalar_t>) const {
        return static_cast<idx_t>(simd::traits<scalar_t>::size);
    }
};

} // anonymous namespace

idx_t DefaultCompute::batch_size(var::scalar_tag tag) const {
    return var::apply_visitor(BatchSize{}, tag);
}

void DefaultCompute::moments(MomentsRef m, Starter const& s, AlgorithmConfig const& ac,
                             OptimizedHamiltonian const& oh) const {
    oh.matrix().match(SelectMatrix{std::move(m), s, ac, oh});
}

}} // namespace cpb::kpm
