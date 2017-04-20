#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

#include <vector>

namespace cpb { namespace kpm {

/**
 Collects moments in the form of simple expectation values:
 `mu_n = <r|Tn(H)|r>` where `bra == ket == r`. It's only
 compatible with the diagonal `calc_moments` algorithms.
 */
template<class scalar_t>
class DiagonalMoments {
public:
    using VectorRef = Ref<VectorX<scalar_t>>;

    DiagonalMoments(idx_t num_moments) : moments(num_moments) {}

    ArrayX<scalar_t>& get() { return moments; }

    /// Number of moments
    idx_t size() const { return moments.size(); }

    /// Zero of the same scalar type as the moments
    constexpr static scalar_t zero() { return scalar_t{0}; }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorRef r0, VectorRef r1);

    /// Collect moments `n` and `n + 1` from the result vectors. Expects `n >= 2`.
    void collect(idx_t n, scalar_t m2, scalar_t m3);

private:
    ArrayX<scalar_t> moments;
    scalar_t m0;
    scalar_t m1;
};

/**
  Moments collector interface for the off-diagonal algorithm.
  Concrete implementations define what part of the KPM vectors
  should be collected and/or apply operators.
 */
template<class scalar_t>
class OffDiagonalMoments {
public:
    using VectorRef = Ref<VectorX<scalar_t>>;

    virtual ~OffDiagonalMoments() = default;

    /// Number of moments
    virtual idx_t size() const = 0;

    /// Collect the first 2 moments which are computer outside the main KPM loop
    virtual void collect_initial(VectorRef r0, VectorRef r1) = 0;

    /// Collect moment `n` from the result vector `r1`. Expects `n >= 2`.
    virtual void collect(idx_t n, VectorRef r1) = 0;
};

/**
 Collects moments in the form of expectation values with an optional operator:
 `mu_n = <beta|op Tn(H)|alpha>` where `beta != alpha`. The `op` can be empty,
 in which case it is not applied.
 */
template<class scalar_t>
class GenericMoments : public OffDiagonalMoments<scalar_t> {
    using VectorRef = typename OffDiagonalMoments<scalar_t>::VectorRef;

public:
    GenericMoments(idx_t num_moments, VectorX<scalar_t> beta, SparseMatrixX<scalar_t> op = {})
        : moments(num_moments), beta(std::move(beta)), op(op.markAsRValue()) {}

    ArrayX<scalar_t>& get() { return moments; }

    idx_t size() const override { return moments.size(); }
    void collect_initial(VectorRef r0, VectorRef r1) override;
    void collect(idx_t n, VectorRef r1) override;

private:
    ArrayX<scalar_t> moments;
    VectorX<scalar_t> beta;
    SparseMatrixX<scalar_t> op;
};

/**
  Collects the computed moments in the form `mu_n = <l|Tn(H)|r>`
  where `l` is a unit vector with `l[i] = 1` and `i` is some
  Hamiltonian index. Multiple `l` vectors can be defined simply
  by defining a vector of Hamiltonian indices `idx` where each
  index is used to form an `l` vector and collect a moment.

  The resulting `data` is a vector of vectors where each outer
  index corresponds to an index from `idx`.
 */
template<class scalar_t>
class MultiUnitCollector : public OffDiagonalMoments<scalar_t> {
    using VectorRef = typename OffDiagonalMoments<scalar_t>::VectorRef;
    using Data = std::vector<ArrayX<scalar_t>>;

public:
    MultiUnitCollector(idx_t num_moments, Indices const& idx)
        : idx(idx), data(idx.cols.size(), ArrayX<scalar_t>(num_moments)) {}

    Data& get() { return data; }

    idx_t size() const override { return data[0].size(); }
    void collect_initial(VectorRef r0, VectorRef r1) override;
    void collect(idx_t n, VectorRef r1) override;

private:
    Indices idx;
    Data data;
};

/**
  Collects vectors of the form `vec_n = op * Tn(H)|r>` into a matrix
  of shape `num_moments * ham_size`. The sparse matrix operator `op`
  is optional.
 */
template<class scalar_t>
class DenseMatrixCollector : public OffDiagonalMoments<scalar_t> {
    using VectorRef = typename OffDiagonalMoments<scalar_t>::VectorRef;

public:
    DenseMatrixCollector(idx_t num_moments, OptimizedHamiltonian const& oh,
                         SparseMatrixX<scalar_t> const& csr_operator = {})
        : data(num_moments, oh.size()), op(csr_operator) {
        if (op.size() != 0) {
            oh.reorder(op);
        }
    }

    MatrixX<scalar_t>& matrix() { return data; }

    idx_t size() const override { return data.rows(); }
    void collect_initial(VectorRef r0, VectorRef r1) override;
    void collect(idx_t n, VectorRef r1) override;

private:
    MatrixX<scalar_t> data;
    SparseMatrixX<scalar_t> op;
};

CPB_EXTERN_TEMPLATE_CLASS(DiagonalMoments)
CPB_EXTERN_TEMPLATE_CLASS(GenericMoments)
CPB_EXTERN_TEMPLATE_CLASS(MultiUnitCollector)
CPB_EXTERN_TEMPLATE_CLASS(DenseMatrixCollector)

}} // namespace cpb::kpm
