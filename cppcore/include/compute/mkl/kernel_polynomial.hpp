#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

namespace tbm { namespace compute {

namespace detail {
    template<class scalar_t> struct mkl_typemap;
    template<> struct mkl_typemap<float> { using type = float; };
    template<> struct mkl_typemap<double> { using type = double; };
    template<> struct mkl_typemap<std::complex<float>> { using type = MKL_Complex8; };
    template<> struct mkl_typemap<std::complex<double>> { using type = MKL_Complex16; };
}

/// Get the corresponding MKL C API type from the C++ type `scalar_t`
template<class scalar_t>
using mkl_t = typename detail::mkl_typemap<scalar_t>::type;

/// Get the MKL csrmv (sparse matrix vector multiplication) function for the C++ type `scalar_t`
template<class scalar_t> struct mkl_xcsrmv;
template<> struct mkl_xcsrmv<float> { static constexpr auto call = mkl_scsrmv; };
template<> struct mkl_xcsrmv<double> { static constexpr auto call = mkl_dcsrmv; };
template<> struct mkl_xcsrmv<std::complex<float>> { static constexpr auto call = mkl_ccsrmv; };
template<> struct mkl_xcsrmv<std::complex<double>> { static constexpr auto call = mkl_zcsrmv; };

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
    scalar_t const alpha = 1;
    scalar_t const beta = -1;

    auto const size = end - start;
    auto const start_idx = matrix.outerIndexPtr()[start];

    using mkl_scalar_t = mkl_t<scalar_t>;
    mkl_xcsrmv<scalar_t>::call(
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
