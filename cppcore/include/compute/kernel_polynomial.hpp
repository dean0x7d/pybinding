#pragma once
#include "compute/detail.hpp"

#ifdef TBM_USE_MKL
# include "mkl/kernel_polynomial.hpp"
#else
# include "eigen3/kernel_polynomial.hpp"
#endif

#include "numeric/ellmatrix.hpp"

namespace tbm { namespace compute {

template<class scalar_t>
inline void kpm_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    for (auto row = start; row < end; ++row) {
        y[row] = -y[row];
    }

    for (auto n = 0; n < matrix.nnz_per_row; ++n) {
        for (auto row = start; row < end; ++row) {
            auto const a = matrix.data(row, n);
            auto const b = x[matrix.indices(row, n)];
            y[row] += detail::mul(a, b);
        }
    }
}

}} // namespace tbm::compute
