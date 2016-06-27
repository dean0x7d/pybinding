#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "numeric/ellmatrix.hpp"
#include "numeric/traits.hpp"

#include "compute/detail.hpp"
#include "detail/macros.hpp"
#include "support/simd.hpp"

#ifdef CPB_USE_MKL
# include "compute/mkl/wrapper.hpp"
#endif

namespace cpb { namespace compute {

/**
 Off-diagonal KPM compute kernel for CSR matrix

 Equivalent to: y = matrix * x - y
 */
#ifndef CPB_USE_MKL

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_kernel(int start, int end, SparseMatrixX<scalar_t> const& matrix,
                VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    auto const data = matrix.valuePtr();
    auto const indices = matrix.innerIndexPtr();
    auto const indptr = matrix.outerIndexPtr();

    for (auto row = start; row < end; ++row) {
        auto r = scalar_t{0};
        for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
            r += detail::mul(data[n], x[indices[n]]);
        }
        y[row] = r - y[row];
    }
}

#else // CPB_USE_MKL

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_kernel(int start, int end, SparseMatrixX<scalar_t> const& matrix,
                VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    if (end <= start) {
        return;
    }

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

#endif // CPB_USE_MKL

/**
 Diagonal KPM compute kernel for CSR matrix

 Equivalent to:
   y = matrix * x - y
   m2 = x^2
   m3 = dot(x, y)
 */
template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_diag_kernel(int start, int end, SparseMatrixX<scalar_t> const& matrix,
                     VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                     scalar_t& m2, scalar_t& m3) {
    kpm_kernel(start, end, matrix, x, y);
    auto const size = end - start;
    m2 += x.segment(start, size).squaredNorm();
    m3 += y.segment(start, size).dot(x.segment(start, size));
}

/**
 Off-diagonal KPM compute kernel for ELLPACK matrix

 Equivalent to: y = matrix * x - y
 */
#if SIMDPP_USE_NULL // generic version

template<class scalar_t> CPB_ALWAYS_INLINE
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

template<class scalar_t, int skip_last_n = 0,
         int step = simd::detail::traits<scalar_t>::size> CPB_ALWAYS_INLINE
simd::split_loop_t<step> kpm_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
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

    for (auto n = 0; n < matrix.nnz_per_row - skip_last_n; ++n) {
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

    return loop;
}

#endif // SIMDPP_USE_NULL

/**
 Diagonal KPM compute kernel for ELLPACK matrix

 Equivalent to:
   y = matrix * x - y
   m2 = x^2
   m3 = dot(x, y)
 */
#if SIMDPP_USE_NULL // generic version

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_diag_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
                     VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                     scalar_t& m2, scalar_t& m3) {
    kpm_kernel(start, end, matrix, x, y);
    auto const size = end - start;
    m2 += x.segment(start, size).squaredNorm();
    m3 += y.segment(start, size).dot(x.segment(start, size));
}

#else // vectorized using SIMD intrinsics

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_diag_kernel(int start, int end, num::EllMatrix<scalar_t> const& matrix,
                     VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                     scalar_t& m2, scalar_t& m3) {
    // Call the regular compute function, but skip the last loop iteration.
    auto const loop = kpm_kernel<scalar_t, 1>(start, end, matrix, x, y);

    // The last iteration will be done here together with the m2 and m3 sums.
    // This saves memory bandwidth by reusing `y` data (`r2`) which is already
    // in a register. While `x` data (`r1`) is not strictly reused, there is good
    // locality between the `b = gather(x)` and `r2 = load(x)` operations which
    // improves the cache hit rate. Overall, this offers a nice speed improvement.
    auto const n = matrix.nnz_per_row - 1;
    auto const data = &matrix.data(0, n);
    auto const indices = &matrix.indices(0, n);

    using simd_register_t = simd::select_vector_t<scalar_t>;
    auto m2_vec = simd::make_float<simd_register_t>(0);
    auto m3_vec = simd::make_float<simd_register_t>(0);

    for (auto row = loop.start; row < loop.peel_end; ++row) {
        auto const r1 = x[row];
        auto const r2 = y[row] + detail::mul(data[row], x[indices[row]]);
        m2 += detail::square(r1);
        m3 += detail::mul(num::conjugate(r2), r1);
        y[row] = r2;
    }
    for (auto row = loop.peel_end; row < loop.vec_end; row += loop.step) {
        auto const a = simd::load<simd_register_t>(data + row);
        auto const b = simd::gather<simd_register_t>(x.data(), indices + row);
        auto const c = simd::load<simd_register_t>(y.data() + row);

        auto const r1 = simd::load<simd_register_t>(x.data() + row);
        auto const r2 = simd::madd_rc<scalar_t>(a, b, c);
        m2_vec = m2_vec + r1 * r1;
        m3_vec = simd::conjugate_madd_rc<scalar_t>(r2, r1, m3_vec);

        simd::store(y.data() + row, r2);
    }
    for (auto row = loop.vec_end; row < loop.end; ++row) {
        auto const r1 = x[row];
        auto const r2 = y[row] + detail::mul(data[row], x[indices[row]]);
        m2 += detail::square(r1);
        m3 += detail::mul(num::conjugate(r2), r1);
        y[row] = r2;
    }

    m2 += simd::reduce_add(m2_vec);
    m3 += simd::reduce_add_rc<scalar_t>(m3_vec);
}

#endif // SIMDPP_USE_NULL
}} // namespace cpb::compute
