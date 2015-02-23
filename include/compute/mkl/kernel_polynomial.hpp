#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

namespace compute {
/**
 KPM computation kernel implemented via MKL functions
 */
template<typename scalar_t>
inline void kpm_kernel(const int size, const SparseMatrixX<scalar_t>& matrix,
                       const VectorX<scalar_t>& x, VectorX<scalar_t>& y);

template<>
inline void kpm_kernel<float>(const int size, const SparseMatrixXf& matrix,
                              const VectorXf& x, VectorXf& y)
{
    const char transa = 'n'; // specifies normal (non-transposed) matrix-vector multiplication
    const char metdescra[8] = "GLNC"; // G - general matrix, C - zero-based indexing, LN - ignored
    const float alpha = 1;
    const float beta = -1;

    mkl_scsrmv(
        &transa,
        &size, &size,
        &alpha,
        metdescra,
        // input matrix
        matrix.valuePtr(),
        matrix.innerIndexPtr(),
        matrix.outerIndexPtr(),
        matrix.outerIndexPtr() + 1,
        // input vector
        x.data(),
        &beta,
        // output vector
        y.data()
    );
}

template<>
inline void kpm_kernel<std::complex<float>>(const int size, const SparseMatrixXcf& matrix,
                                            const VectorXcf& x, VectorXcf& y)
{
    const char transa = 'n'; // specifies normal (non-transposed) matrix-vector multiplication
    const char metdescra[8] = "GLNC"; // G - general matrix, C - zero-based indexing, LN - ignored
    const MKL_Complex8 alpha{1, 0};
    const MKL_Complex8 beta{-1, 0};

    mkl_ccsrmv(
        &transa,
        &size, &size,
        &alpha,
        metdescra,
        // input matrix - cast from complex<float>* to MKL_Complex8*
        reinterpret_cast<const MKL_Complex8*>(matrix.valuePtr()),
        matrix.innerIndexPtr(),
        matrix.outerIndexPtr(),
        matrix.outerIndexPtr() + 1,
        // input vector
        reinterpret_cast<const MKL_Complex8*>(x.data()),
        &beta,
        // output vector
        reinterpret_cast<MKL_Complex8*>(y.data())
    );
}

} // namespace compute
