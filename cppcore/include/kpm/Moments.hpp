#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

namespace cpb { namespace kpm {

/**
 Collects moments in the form of simple expectation values:
 `mu_n = <r|Tn(H)|r>` where `bra == ket == r`. It's only
 compatible with the diagonal `calc_moments` algorithms.
 */
struct DiagonalMoments {
    idx_t num_moments;
    var::Complex<ArrayX> data;

    DiagonalMoments(idx_t num_moments) : num_moments(num_moments) {}
};

struct BatchDiagonalMoments {
    idx_t num_moments;
    idx_t batch_size;
    var::Complex<ArrayXX> data;

    BatchDiagonalMoments(idx_t num_moments, idx_t batch_size)
        : num_moments(num_moments), batch_size(batch_size) {}
};

/**
 Collects moments in the form of expectation values with an optional operator:
 `mu_n = <beta|op Tn(H)|alpha>` where `beta != alpha`. The `op` can be empty,
 in which case it is not applied.
 */
struct GenericMoments {
    idx_t num_moments;
    VectorXcd const& alpha;
    VectorXcd const& beta;
    SparseMatrixXcd const& op;
    var::Complex<ArrayX> data;

    GenericMoments(idx_t num_moments, VectorXcd const& alpha, VectorXcd const& beta,
                   SparseMatrixXcd const& op)
        : num_moments(num_moments), alpha(alpha), beta(beta), op(op) {}
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
struct MultiUnitMoments {
    template<class scalar_t> using Data = std::vector<ArrayX<scalar_t>>;

    idx_t num_moments;
    Indices const& idx;
    var::Complex<Data> data;

    MultiUnitMoments(idx_t num_moments, Indices const& idx)
        : num_moments(num_moments), idx(idx) {}
};

/**
  Collects vectors of the form `vec_n = op * Tn(H)|r>` into a matrix
  of shape `num_moments * ham_size`. The sparse matrix operator `op`
  is optional.
 */
struct DenseMatrixMoments {
    idx_t num_moments;
    VariantCSR op;
    var::Complex<MatrixX> data;

    DenseMatrixMoments(idx_t num_moments, VariantCSR op = {})
        : num_moments(num_moments), op(std::move(op)) {}
};

/**
 Adds up moments for the stochastic KPM procedure
 */
struct MomentAccumulator {
    idx_t num_moments;
    idx_t total; ///< add new result to existing data for this number of moments
    idx_t batch_size;
    idx_t _count = 0; ///< internal: keeps track of how many moments have been summed up so far
    var::Complex<ArrayX> data;

    MomentAccumulator(idx_t num_moments, idx_t total, idx_t batch_size = 1)
        : num_moments(num_moments), total(total), batch_size(batch_size) {}

    void add(var::Complex<ArrayX> const& other);
    void add(var::Complex<ArrayXX> const& other);
};

struct MomentMultiplication {
    var::Complex<MatrixX> data;

    MomentMultiplication(idx_t num_moments, var::scalar_tag tag);

    void matrix_mul_add(DenseMatrixMoments const& a, DenseMatrixMoments const& b);
    void normalize(idx_t total);
};

using MomentsRef = var::variant<DiagonalMoments*, BatchDiagonalMoments*, GenericMoments*,
                                MultiUnitMoments*, DenseMatrixMoments*>;

template<class M>
void apply_damping(M& moments, Kernel const& kernel) {
    var::apply_visitor(kernel, moments.data);
}

struct ExtractData {
    idx_t num_moments;

    template<class scalar_t>
    ArrayXcd operator()(ArrayX<scalar_t> const& data) const {
        return data.template cast<std::complex<double>>().head(num_moments);
    }
};

template<class M>
ArrayXcd extract_data(M const& moments, idx_t num_moments) {
    return moments.data.match(ExtractData{num_moments});
}

/// Return the velocity operator for the direction given by the `alpha` position vector
VariantCSR velocity(Hamiltonian const& hamiltonian, ArrayXf const& alpha);

}} // namespace cpb::kpm
