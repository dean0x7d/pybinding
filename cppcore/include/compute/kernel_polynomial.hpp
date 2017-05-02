#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "numeric/ellmatrix.hpp"
#include "numeric/traits.hpp"

#include "compute/detail.hpp"
#include "detail/macros.hpp"
#include "support/simd.hpp"

namespace cpb { namespace compute {

/**
 KPM-specialized sparse matrix-vector multiplication (CSR, off-diagonal)

 Equivalent to: y = matrix * x - y
 */
template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
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

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
              MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y) {
    auto const data = matrix.valuePtr();
    auto const indices = matrix.innerIndexPtr();
    auto const indptr = matrix.outerIndexPtr();

    using Row = Eigen::Matrix<scalar_t, 1, Eigen::Dynamic>;
    auto tmp = Row(x.cols());
    for (auto row = start; row < end; ++row) {
        tmp.setZero();
        for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
            tmp += data[n] * x.row(indices[n]);
        }
        y.row(row) = tmp - y.row(row);
    }
}

/**
 KPM-specialized sparse matrix-vector multiplication (CSR, diagonal)

 Equivalent to:
   y = matrix * x - y
   m2 = x^2
   m3 = dot(x, y)
 */
template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                       scalar_t& m2, scalar_t& m3) {
    kpm_spmv(start, end, matrix, x, y);
    auto const size = end - start;
    m2 += x.segment(start, size).squaredNorm();
    m3 += y.segment(start, size).dot(x.segment(start, size));
}

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, SparseMatrixX<scalar_t> const& matrix,
                       MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                       simd::array<scalar_t>& m2, simd::array<scalar_t>& m3) {
    kpm_spmv(start, end, matrix, x, y);
    auto const size = end - start;
    auto const cols = x.cols();
    for (auto i = 0; i < cols; ++i) {
        m2[i] += x.col(i).segment(start, size).squaredNorm();
        m3[i] += y.col(i).segment(start, size).dot(x.col(i).segment(start, size));
    }
}

/**
 KPM-specialized sparse matrix-vector multiplication (ELLPACK, off-diagonal)

 Equivalent to: y = matrix * x - y
 */
#if SIMDPP_USE_NULL // generic version

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
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

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
              MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y) {
    for (auto row = start; row < end; ++row) {
        y.row(row) = -y.row(row);
    }

    for (auto n = 0; n < matrix.nnz_per_row; ++n) {
        for (auto row = start; row < end; ++row) {
            y.row(row) += matrix.data(row, n) * x.row(matrix.indices(row, n));
        }
    }
}

#else // vectorized using SIMD intrinsics

template<class scalar_t, idx_t skip_last_n = 0,
         idx_t step = simd::traits<scalar_t>::size> CPB_ALWAYS_INLINE
simd::split_loop_t<step> kpm_spmv(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
                                  VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    using simd_register_t = simd::select_vector_t<scalar_t>;
    auto const loop = simd::split_loop(y.data(), start, end);

    auto const px0 = x.data();
    for (auto n = 0; n < matrix.nnz_per_row - skip_last_n; ++n) {
        auto data = &matrix.data(0, n) + start;
        auto idx = &matrix.indices(0, n) + start;
        auto py = y.data() + start;

        for (auto _ = loop.start; _ < loop.peel_end; ++_, ++data, ++idx, ++py) {
            auto const a = *data;
            auto const b = px0[*idx];
            auto const c = (n == 0) ? -*py : *py;
            *py = detail::mul(a, b) + c;
        }
        for (auto _ = loop.peel_end; _ < loop.vec_end;
             _ += step, data += step, idx += step, py += step) {
            auto const a = simd::load<simd_register_t>(data);
            auto const b = simd::gather<simd_register_t>(px0, idx);
            auto c = simd::load<simd_register_t>(py);
            if (n == 0) { c = simd::neg(c); }
            simd::store(py, simd::madd_rc<scalar_t>(a, b, c));
        }
        for (auto _ = loop.vec_end; _ < loop.end; ++_, ++data, ++idx, ++py) {
            auto const a = *data;
            auto const b = px0[*idx];
            auto const c = (n == 0) ? -*py : *py;
            *py = detail::mul(a, b) + c;
        }
    }

    return loop;
}

template<class scalar_t, idx_t skip_last_n = 0> CPB_ALWAYS_INLINE
void kpm_spmv(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
              MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y) {
    using simd_register_t = simd::select_vector_t<scalar_t>;
    static constexpr auto step = simd::traits<scalar_t>::size;

    auto const px0 = x.data();
    for (auto n = 0; n < matrix.nnz_per_row - skip_last_n; ++n) {
        auto data = &matrix.data(0, n) + start;
        auto data_end = &matrix.data(0, n) + end;
        auto idx = &matrix.indices(0, n) + start;
        auto py = y.data() + start * step;

        for (; data < data_end; ++data, ++idx, py += step) {
            auto const a = simd::load_splat_rc<simd_register_t>(data);
            auto const b = simd::load<simd_register_t>(px0 + *idx * step);
            auto c = simd::load<simd_register_t>(py);
            if (n == 0) { c = simd::neg(c); }
            simd::store(py, simd::madd_rc<scalar_t>(a, b, c));
        }
    }
}

#endif // SIMDPP_USE_NULL

/**
 KPM-specialized sparse matrix-vector multiplication (ELLPACK, diagonal)

 Equivalent to:
   y = matrix * x - y
   m2 = x^2
   m3 = dot(x, y)
 */
#if SIMDPP_USE_NULL // generic version

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                       scalar_t& m2, scalar_t& m3) {
    kpm_spmv(start, end, matrix, x, y);
    auto const size = end - start;
    m2 += x.segment(start, size).squaredNorm();
    m3 += y.segment(start, size).dot(x.segment(start, size));
}

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
                       MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                       simd::array<scalar_t>& m2, simd::array<scalar_t>& m3) {
    kpm_spmv(start, end, matrix, x, y);
    auto const size = end - start;
    auto const cols = x.cols();
    for (auto i = 0; i < cols; ++i) {
        m2[i] += x.col(i).segment(start, size).squaredNorm();
        m3[i] += y.col(i).segment(start, size).dot(x.col(i).segment(start, size));
    }
}

#else // vectorized using SIMD intrinsics

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y,
                       scalar_t& m2, scalar_t& m3) {
    // Call the regular compute function, but skip the last loop iteration.
    auto const loop = kpm_spmv<scalar_t, 1>(start, end, matrix, x, y);

    // The last iteration will be done here together with the m2 and m3 sums.
    // This saves memory bandwidth by reusing `y` data (`r2`) which is already
    // in a register. While `x` data (`r1`) is not strictly reused, there is good
    // locality between the `b = gather(x)` and `r2 = load(x)` operations which
    // improves the cache hit rate. Overall, this offers a nice speed improvement.
    using simd_register_t = simd::select_vector_t<scalar_t>;
    static constexpr auto step = simd::traits<scalar_t>::size;
    auto const n = matrix.nnz_per_row - 1;

    auto data = &matrix.data(0, n) + start;
    auto idx = &matrix.indices(0, n) + start;
    auto px0 = x.data();
    auto px = x.data() + start;
    auto py = y.data() + start;
    auto m2_vec = simd::make_float<simd_register_t>(0);
    auto m3_vec = simd::make_float<simd_register_t>(0);

    for (auto _ = loop.start; _ < loop.peel_end; ++_, ++data, ++idx, ++py, ++px) {
        auto const a = *data;
        auto const b = px0[*idx];
        auto const c = (n == 0 ? -*py : *py);

        auto const r1 = *px;
        auto const r2 = a * b + c;
        m2 += detail::square(r1);
        m3 += detail::mul(num::conjugate(r2), r1);

        *py = r2;
    }
    for (auto _ = loop.peel_end; _ < loop.vec_end;
         _ += step, data += step, idx += step, py += step, px += step) {
        auto const a = simd::load<simd_register_t>(data);
        auto const b = simd::gather<simd_register_t>(px0, idx);
        auto c = simd::load<simd_register_t>(py);
        if (n == 0) { c = simd::neg(c); }

        auto const r1 = simd::load<simd_register_t>(px);
        auto const r2 = simd::madd_rc<scalar_t>(a, b, c);
        m2_vec = m2_vec + r1 * r1;
        m3_vec = simd::conjugate_madd_rc<scalar_t>(r2, r1, m3_vec);

        simd::store(py, r2);
    }
    for (auto _ = loop.vec_end; _ < loop.end; ++_, ++data, ++idx, ++py, ++px) {
        auto const a = *data;
        auto const b = px0[*idx];
        auto const c = (n == 0 ? -*py : *py);

        auto const r1 = *px;
        auto const r2 = a * b + c;
        m2 += detail::square(r1);
        m3 += detail::mul(num::conjugate(r2), r1);

        *py = r2;
    }

    m2 += simd::reduce_add(m2_vec);
    m3 += simd::reduce_add_rc<scalar_t>(m3_vec);
}

template<class scalar_t> CPB_ALWAYS_INLINE
void kpm_spmv_diagonal(idx_t start, idx_t end, num::EllMatrix<scalar_t> const& matrix,
                       MatrixX<scalar_t> const& x, MatrixX<scalar_t>& y,
                       simd::array<scalar_t>& m2, simd::array<scalar_t>& m3) {
    kpm_spmv<scalar_t, 1>(start, end, matrix, x, y);

    using simd_register_t = simd::select_vector_t<scalar_t>;
    static constexpr auto step = simd::traits<scalar_t>::size;
    auto const n = matrix.nnz_per_row - 1;

    auto data = &matrix.data(0, n) + start;
    auto data_end = &matrix.data(0, n) + end;
    auto idx = &matrix.indices(0, n) + start;
    auto px0 = x.data();
    auto px = x.data() + start * step;
    auto py = y.data() + start * step;
    auto m2_vec = simd::make_float<simd_register_t>(0);
    auto m3_vec = simd::make_float<simd_register_t>(0);

    for (; data < data_end; ++data, ++idx, px += step, py += step) {
        auto const a = simd::load_splat_rc<simd_register_t>(data);
        auto const b = simd::load<simd_register_t>(px0 + *idx * step);
        auto c = simd::load<simd_register_t>(py);
        if (n == 0) { c = simd::neg(c); }

        auto const r1 = simd::load<simd_register_t>(px);
        auto const r2 = simd::madd_rc<scalar_t>(a, b, c);
        m2_vec = m2_vec + r1 * r1;
        m3_vec = simd::conjugate_madd_rc<scalar_t>(r2, r1, m3_vec);

        simd::store(py, r2);
    }

    m2_vec = simd::reduce_imag<scalar_t>(m2_vec);
    simd::store_u(m2.data(), simd::load_u<simd_register_t>(m2.data()) + m2_vec);
    simd::store_u(m3.data(), simd::load_u<simd_register_t>(m3.data()) + m3_vec);
}

#endif // SIMDPP_USE_NULL
}} // namespace cpb::compute
