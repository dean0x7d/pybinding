#pragma once
#include "kpm/OptimizedHamiltonian.hpp"
#include "detail/macros.hpp"

#include "support/simd.hpp"

namespace cpb { namespace kpm {

template<class scalar_t>
class DiagonalCollector {
public:
    using Vector = VectorX<scalar_t>;
    using VectorRef = Eigen::Ref<Vector>;

    ArrayX<scalar_t> moments;
    scalar_t m0;
    scalar_t m1;

    DiagonalCollector(idx_t num_moments) : moments(num_moments) {}

    idx_t size() const { return moments.size(); }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void initial(VectorRef r0, VectorRef r1);

    /// Collect moments `n` and `n + 1` from the result vectors. Expects `n >= 2`.
    void operator()(idx_t n, scalar_t m2, scalar_t m3);

    /// Zero of the same scalar type as the moments
    static constexpr scalar_t zero() { return scalar_t{0}; }
};

template<class scalar_t>
class BatchDiagonalCollector {
public:
    using Vector = MatrixX<scalar_t>;
    using VectorRef = Eigen::Ref<Vector>;

    ArrayXX<scalar_t> moments;
    simd::array<scalar_t> m0;
    simd::array<scalar_t> m1;

    BatchDiagonalCollector(idx_t num_moments, idx_t batch_size)
        : moments(num_moments, batch_size) {}

    idx_t size() const { return moments.rows(); }
    void initial(VectorRef r0, VectorRef r1);
    void operator()(idx_t n, simd::array<scalar_t> m2, simd::array<scalar_t> m3);
    static constexpr simd::array<scalar_t> zero() { return {{0}}; }
};


/**
 Moments collector interface for the off-diagonal algorithm.
 Concrete implementations define what part of the KPM vectors
 should be collected and/or apply operators.
 */
template<class scalar_t>
class OffDiagonalCollector {
public:
    using Vector = VectorX<scalar_t>;
    using VectorRef = Eigen::Ref<Vector>;

    virtual ~OffDiagonalCollector() = default;

    /// Number of moments
    virtual idx_t size() const = 0;

    /// Collect the first 2 moments which are computer outside the main KPM loop
    virtual void initial(VectorRef r0, VectorRef r1) = 0;

    /// Collect moment `n` from the result vector `r1`. Expects `n >= 2`.
    virtual void operator()(idx_t n, VectorRef r1) = 0;
};

template<class scalar_t>
class GenericCollector : public OffDiagonalCollector<scalar_t> {
    using VectorRef = typename OffDiagonalCollector<scalar_t>::VectorRef;

public:
    ArrayX<scalar_t> moments;
    VectorX<scalar_t> beta;
    SparseMatrixX<scalar_t> op;

    GenericCollector(idx_t num_moments, OptimizedHamiltonian const& oh, VectorXcd const& alpha_,
                     VectorXcd const& beta_, SparseMatrixXcd const& op_);

    idx_t size() const override { return moments.size(); }
    void initial(VectorRef r0, VectorRef r1) override;
    void operator()(idx_t n, VectorRef r1) override;
};

template<class scalar_t>
class MultiUnitCollector : public OffDiagonalCollector<scalar_t> {
    using VectorRef = typename OffDiagonalCollector<scalar_t>::VectorRef;

public:
    Indices const& idx;
    std::vector<ArrayX<scalar_t>> moments;

    MultiUnitCollector(idx_t num_moments, Indices const& idx)
        : idx(idx), moments(idx.dest.size(), ArrayX<scalar_t>(num_moments)) {}

    idx_t size() const override { return moments[0].size(); }
    void initial(VectorRef r0, VectorRef r1) override;
    void operator()(idx_t n, VectorRef r1) override;
};

template<class scalar_t>
class DenseMatrixCollector : public OffDiagonalCollector<scalar_t> {
    using VectorRef = typename OffDiagonalCollector<scalar_t>::VectorRef;

public:
    SparseMatrixX<scalar_t> op;
    MatrixX<scalar_t> moments;

    DenseMatrixCollector(idx_t num_moments, OptimizedHamiltonian const& oh,
                         VariantCSR const& op_);

    idx_t size() const override { return moments.rows(); }
    void initial(VectorRef r0, VectorRef r1) override;
    void operator()(idx_t n, VectorRef r1) override;
};

CPB_EXTERN_TEMPLATE_CLASS(DiagonalCollector)
CPB_EXTERN_TEMPLATE_CLASS(BatchDiagonalCollector)
CPB_EXTERN_TEMPLATE_CLASS(GenericCollector)
CPB_EXTERN_TEMPLATE_CLASS(MultiUnitCollector)
CPB_EXTERN_TEMPLATE_CLASS(DenseMatrixCollector)

}} // namespace cpb::kpm
