#pragma once
#include "compute/detail.hpp"

#ifdef TBM_USE_MKL
# include "mkl/kernel_polynomial.hpp"
#else
# include "eigen3/kernel_polynomial.hpp"
#endif

#include "numeric/ellmatrix.hpp"
#include "support/simd.hpp"

namespace tbm { namespace compute {
#if SIMDPP_USE_NULL // generic version

template<class scalar_t> TBM_ALWAYS_INLINE
void kpm_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
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

#else // vectorized using SIMD intrinsics

template<class scalar_t> TBM_ALWAYS_INLINE
void kpm_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
                VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    using simd_register_t = simd::select_vector_t<scalar_t>;
    auto const loop = simd::split_loop(y.data(), start, end);

    for (auto row = loop.start; row < loop.peel_end; ++row) {
        y[row] = -y[row];
    }
    for (auto row = loop.peel_end; row < loop.vec_end; row += loop.step) {
        auto const r = simd::load<simd_register_t>(y.data() + row);
        simd::store(y.data() + row, simd::neg(r));
    }
    for (auto row = loop.vec_end; row < loop.end; ++row) {
        y[row] = -y[row];
    }

    for (auto n = 0; n < matrix.nnz_per_row; ++n) {
        auto const data = &matrix.data(0, n);
        auto const indices = &matrix.indices(0, n);

        for (auto row = loop.start; row < loop.peel_end; ++row) {
            y[row] += detail::mul(data[row], x[indices[row]]);
        }
        for (auto row = loop.peel_end; row < loop.vec_end; row += loop.step) {
            auto const a = simd::load<simd_register_t>(data + row);
            auto const b = simd::gather<simd_register_t>(x.data(), indices + row);
            auto const c = simd::load<simd_register_t>(y.data() + row);
            simd::store(y.data() + row, simd::madd_rc<scalar_t>(a, b, c));
        }
        for (auto row = loop.vec_end; row < loop.end; ++row) {
            y[row] += detail::mul(data[row], x[indices[row]]);
        }
    }
}

#endif // SIMDPP_USE_NULL
}} // namespace tbm::compute
