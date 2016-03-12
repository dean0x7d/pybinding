#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "compute/mkl/wrapper.hpp"

namespace tbm { namespace compute {

/**
 KPM computation kernel implemented via MKL functions
 */
template<class scalar_t>
inline void kpm_kernel(int start, int end, SparseMatrixX<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    if (end <= start)
        return;

    char const transa = 'n'; // specifies normal (non-transposed) matrix-vector multiplication
    char const metdescra[8] = "GLNC"; // G - general matrix, C - zero-based indexing, LN - ignored
    auto const alpha = scalar_t{1};
    auto const beta = scalar_t{-1};

    auto const size = end - start;
    auto const start_idx = matrix.outerIndexPtr()[start];

    using mkl_scalar_t = mkl::type<scalar_t>;
    mkl::csrmv<scalar_t>::call(
        &transa,
        &size, &size,
        reinterpret_cast<mkl_scalar_t const*>(&alpha),
        metdescra,
        // input matrix
        reinterpret_cast<mkl_scalar_t const*>(matrix.valuePtr()) + start_idx,
        matrix.innerIndexPtr() + start_idx,
        matrix.outerIndexPtr() + start,
        matrix.outerIndexPtr() + 1 + start,
        // input vector
        reinterpret_cast<mkl_scalar_t const*>(x.data()),
        reinterpret_cast<mkl_scalar_t const*>(&beta),
        // output vector
        reinterpret_cast<mkl_scalar_t*>(y.data()) + start
    );
}

}} // namespace tbm::compute
